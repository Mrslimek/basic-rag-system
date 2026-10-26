"""
THE GRAPH branch of the single LlamaIndex pipeline — property-graph retrieval
parallel to the vector pipeline in app/llama_rag.py — plus the hybrid branch
(vector-first Qdrant chunks with Neo4j graph facts appended to their text);
both live here so app/llama_rag.py stays Neo4j-free at import time.

  Postgres  = source of truth for chunk text (unchanged, worker-owned).
  Neo4j     = PropertyGraph store written ONLY by PropertyGraphIndex: chunk
              nodes (__Node__/Chunk), entity nodes (__Node__/__Entity__),
              MENTIONS edges plus LLM-extracted typed relations. Triplet
              extraction happens inside the index via SchemaLLMPathExtractor —
              nothing hand-builds the graph.
  Retrieval = LLMSynonymRetriever (lexical graph walk) + VectorContextRetriever
              (Neo4j-native "entity" vector index, same 384-dim e5 model),
              fused by PropertyGraphIndex's retriever with text fallback; the
              hybrid branch instead walks MENTIONS chunks -> entities and one
              semantic step beyond, appending "Graph facts" to each chunk.

Schema gotcha that must survive any refactor: SchemaLLMPathExtractor uppercases
every entity/relation type before Literal validation, so lowercase Literals
silently extract zero triplets — ENTITIES/RELATIONS/VALIDATION below are
uppercase for that reason, not style.

Run the graph rebuild after corpus changes (mass LLM calls):
    uv run python -m app.graph_rag sync
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Literal

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import EntityNode, Triplet
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    LLMSynonymRetriever,
    PropertyGraphIndex,
    SchemaLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import get_embedding_model
from app.llama_rag import FILTER_SERVICE_KEY, METADATA_KEYS, build_index
from app.llm import get_chat_model
from app.models import Service, ServiceDocumentationChunk

logger = logging.getLogger("graph_rag")
settings = get_settings()

# Uppercase literals — see module docstring, this is a hard extractor constraint.
ENTITIES = Literal["SERVICE", "ENDPOINT", "DATABASE", "QUEUE", "PROTOCOL", "CONCEPT"]
RELATIONS = Literal["CALLS", "AUTHENTICATES_VIA", "STORES_IN", "PUBLISHES_TO", "USES", "PROVIDES"]
VALIDATION = [
    ("SERVICE", "PROVIDES", "ENDPOINT"),
    ("SERVICE", "CALLS", "SERVICE"),
    ("SERVICE", "STORES_IN", "DATABASE"),
    ("SERVICE", "PUBLISHES_TO", "QUEUE"),
    ("SERVICE", "USES", "PROTOCOL"),
    ("SERVICE", "USES", "SERVICE"),
    ("SERVICE", "USES", "ENDPOINT"),
    ("SERVICE", "AUTHENTICATES_VIA", "PROTOCOL"),
    ("ENDPOINT", "USES", "PROTOCOL"),
]


_graph_store: Neo4jPropertyGraphStore | None = None


def get_graph_store() -> Neo4jPropertyGraphStore:
    """Process-wide singleton connection to Neo4j (constraints + entity vector
    index are created idempotently by the store itself on construction)."""
    global _graph_store
    if _graph_store is None:
        _graph_store = Neo4jPropertyGraphStore(
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            url=settings.NEO4J_URI,
            database="neo4j",
        )
    return _graph_store


def build_kg_extractors() -> list:
    """Schema-constrained LLM extractor + the free-form implicit one, so MENTIONS
    edges and any schema-valid triplet both land in the graph."""
    return [
        SchemaLLMPathExtractor(
            llm=get_chat_model(temperature=0.0, max_tokens=1024),
            possible_entities=ENTITIES,
            possible_relations=RELATIONS,
            kg_validation_schema=VALIDATION,
            # 4 triplets ≈ 500 output tokens, inside the extractor's 512 cap.
            max_triplets_per_chunk=4,
            strict=True,
            allow_additional_properties=False,
            # 1.5B sometimes emits truncated/invalid JSON (observed: EOF while
            # parsing object); one bad chunk must not abort the whole sync.
            raise_on_error=False,
        ),
        ImplicitPathExtractor(),
    ]


def build_graph_index() -> PropertyGraphIndex:
    """Index over the live Neo4j property graph — a derived view; Postgres owns
    the text, the graph is rebuilt wholesale by sync_graph_from_postgres."""
    return PropertyGraphIndex.from_existing(
        get_graph_store(),
        llm=get_chat_model(),
        embed_model=get_embedding_model(),
        kg_extractors=build_kg_extractors(),
        use_async=True,
    )


def build_graph_retriever(top_k: int = 4) -> BaseRetriever:
    """No vector_store kwarg: VectorContextRetriever queries Neo4j's native
    "entity" vector index where embed_kg_nodes stored entity embeddings."""
    store = get_graph_store()
    return build_graph_index().as_retriever(
        sub_retrievers=[
            LLMSynonymRetriever(
                graph_store=store,
                llm=get_chat_model(),
                include_text=True,
                path_depth=1,
                limit=max(30, top_k * 10),
            ),
            VectorContextRetriever(
                graph_store=store,
                embed_model=get_embedding_model(),
                include_text=True,
                similarity_top_k=top_k,
                path_depth=1,
                limit=30,
            ),
        ],
        include_text=True,
    )


# Hop 1 of the hybrid expansion: chunk -> MENTIONS -> entity, resolving the
# entity's display name + type label. Hand-written Cypher because no store
# primitive walks edges from chunk ids; only the $ids param is injectable.
_MENTIONS_HOP_QUERY = """
MATCH (c:Chunk)-[:MENTIONS]->(e:__Entity__)
WHERE c.id IN $ids
RETURN c.id AS chunk_id, e.name AS name,
       [l IN labels(e) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0] AS type
"""


@dataclass(frozen=True)
class _EntityMention:
    """One mentioned entity of a chunk: display name + its type label."""

    name: str
    label: str


def build_hybrid_retriever(
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
) -> BaseRetriever:
    """Hybrid branch: stage 1 reuses the Qdrant retriever exactly like the
    vector branch (same top_k, same service_filter semantics); the Neo4j
    expansion runs at retrieve time inside _HybridRetriever, so a down Neo4j
    degrades to plain vector results instead of failing at engine build."""
    filters = (
        MetadataFilters(
            filters=[MetadataFilter(key=FILTER_SERVICE_KEY, value=service_filter)]
        )
        if service_filter
        else None
    )
    vector_retriever = build_index(collection_name).as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )
    return _HybridRetriever(vector_retriever=vector_retriever, top_k=top_k)


class _HybridRetriever(BaseRetriever):
    """Vector-first hybrid: the Qdrant chunks from stage 1 stay the sources —
    stages 2-3 only append graph facts to their text. Any Neo4j error during
    the expansion degrades to the plain vector results (one warning per
    retrieve call)."""

    def __init__(self, vector_retriever: BaseRetriever, top_k: int) -> None:
        super().__init__()
        self._vector_retriever = vector_retriever
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = self._vector_retriever.retrieve(query_bundle)
        try:
            # get_graph_store() is contacted here, not at construction: an
            # unreachable Neo4j must degrade hybrid to the vector results.
            facts_by_chunk = _expand_graph_facts(
                get_graph_store(),
                [node.node_id for node in nodes],
                self._top_k,
            )
        except Exception as exc:
            logger.warning(
                f"Neo4j unavailable during hybrid expansion ({exc}); "
                "returning plain vector results."
            )
            return nodes
        if not facts_by_chunk:
            return nodes
        return _attach_graph_facts(nodes, facts_by_chunk)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # The sync Qdrant client and the Neo4j driver both block; to_thread
        # keeps them off the event loop (same pattern as the graph sync).
        return await asyncio.to_thread(self._retrieve, query_bundle)


def _expand_graph_facts(
    store: Neo4jPropertyGraphStore,
    chunk_ids: list[str],
    top_k: int,
) -> dict[str, str]:
    """Stages 2-3 per retrieved chunk: hop 1 resolves the mentioned entities
    over MENTIONS edges, hop 2 walks one semantic step from those entities,
    and each chunk's result is packaged as its "Graph facts" text block.
    Chunks without mentions are absent from the returned mapping."""
    rows = store.structured_query(_MENTIONS_HOP_QUERY, param_map={"ids": chunk_ids})
    mentions_by_chunk: dict[str, list[_EntityMention]] = {}
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        mention = _EntityMention(name=name, label=row.get("type") or "ENTITY")
        bucket = mentions_by_chunk.setdefault(str(row["chunk_id"]), [])
        if mention not in bucket:
            bucket.append(mention)

    facts: dict[str, str] = {}
    for chunk_id, mentions in mentions_by_chunk.items():
        facts[chunk_id] = _format_graph_facts(
            mentions,
            _semantic_triplets(store, mentions, top_k),
        )
    return facts


def _semantic_triplets(
    store: Neo4jPropertyGraphStore,
    mentions: list[_EntityMention],
    top_k: int,
) -> list[Triplet]:
    """Hop 2, via the store's own get_rel_map: its path predicate already
    excludes MENTIONS from the walk, ignore_rels repeats the same exclusion
    in the Python post-filter, and Chunk-labelled endpoints are dropped so
    the walk never returns to chunk text."""
    triplets = store.get_rel_map(
        [EntityNode(name=m.name, label=m.label) for m in mentions],
        depth=1,
        limit=max(30, top_k * 10),
        ignore_rels=["MENTIONS"],
    )
    return [t for t in triplets if t[0].label != "Chunk" and t[2].label != "Chunk"]


def _format_graph_facts(
    mentions: list[_EntityMention], triplets: list[Triplet]
) -> str:
    """The chunk's facts block: mentioned entities with their type labels,
    then deduplicated triplets in the same "X -> REL -> Y" text convention as
    the graph branch (_get_nodes_with_score in sub_retrievers/base.py)."""
    lines = [f"- Entities: {', '.join(f'{m.name} ({m.label})' for m in mentions)}"]
    seen: set[tuple[str, str, str]] = set()
    for source, relation, target in triplets:
        key = (source.id, relation.id, target.id)
        if key not in seen:
            seen.add(key)
            lines.append(f"- {source.id} -> {relation.id} -> {target.id}")
    return "Graph facts:\n" + "\n".join(lines)


def _attach_graph_facts(
    nodes: list[NodeWithScore], facts_by_chunk: dict[str, str]
) -> list[NodeWithScore]:
    """Stage 3 packaging: same id_/metadata/score as the vector node, text =
    original chunk + appended facts block; chunks without facts pass through
    untouched."""
    enriched: list[NodeWithScore] = []
    for node_with_score in nodes:
        facts = facts_by_chunk.get(node_with_score.node.node_id)
        if not facts:
            enriched.append(node_with_score)
            continue
        enriched.append(
            NodeWithScore(
                node=TextNode(
                    id_=node_with_score.node.node_id,
                    text=f"{node_with_score.node.get_content()}\n\n{facts}",
                    metadata=dict(node_with_score.node.metadata),
                ),
                score=node_with_score.score,
            )
        )
    return enriched


async def _count(query: str) -> int:
    """astructured_query returns a list of plain dicts (sanitize_query_output)."""
    rows = await get_graph_store().astructured_query(query)
    return int(next(iter(rows[0].values()))) if rows else 0


async def sync_graph_from_postgres() -> dict[str, int]:
    """
    Full rebuild of the library-owned graph namespace.

    Wipe is scoped to the __Node__ label, which PropertyGraphIndex puts on both
    chunk and entity nodes — the legacy handwritten Entity/Alias/REL labels are
    not touched here (removed in a later wave). Node id = chunk UUID keeps the
    cross-store join key identical to the Qdrant pipeline.
    """
    await get_graph_store().astructured_query("MATCH (n:__Node__) DETACH DELETE n")

    async with get_db_context() as session:
        rows = (
            await session.execute(
                select(
                    ServiceDocumentationChunk.id,
                    ServiceDocumentationChunk.content,
                    ServiceDocumentationChunk.chunk_index,
                    Service.name,
                    Service.doc_file_path,
                    Service.id.label("service_id"),
                )
                .join(Service, Service.id == ServiceDocumentationChunk.service_id)
                .order_by(Service.name, ServiceDocumentationChunk.chunk_index)
            )
        ).all()

    nodes = [
        TextNode(
            id_=str(row.id),
            text=row.content,
            metadata={
                "service_name": row.name,
                "service_id": str(row.service_id),
                "doc_file_path": row.doc_file_path,
                "chunk_index": row.chunk_index,
            },
            excluded_embed_metadata_keys=list(METADATA_KEYS),
            # SchemaLLMPathExtractor reads text via MetadataMode.LLM; without
            # this the service_id UUID and doc_file_path land in the extraction
            # prompt and the model extracts them as entities.
            excluded_llm_metadata_keys=list(METADATA_KEYS),
        )
        for row in rows
    ]

    index = build_graph_index()
    # insert_nodes (not ainsert_nodes): with use_async=True its internals call
    # asyncio.run(), which raises inside this already-running loop; to_thread
    # gives it a thread with no event loop, keeping extraction concurrent.
    await asyncio.to_thread(index.insert_nodes, nodes)

    stats = {
        "postgres_rows": len(rows),
        "graph_chunks": await _count("MATCH (c:Chunk) RETURN count(c) AS n"),
        "graph_entities": await _count("MATCH (e:__Entity__) RETURN count(e) AS n"),
    }
    logger.info(f"Graph rebuilt: {stats}")
    return stats


async def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = await sync_graph_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.graph_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
