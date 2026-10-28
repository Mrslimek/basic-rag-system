"""
THE HYBRID branch of the single LlamaIndex pipeline — vector-first composition.

Stage 1 reuses the Qdrant vector stage from app/vector_rag.py exactly like the
vector branch (same top_k, same service_filter semantics); the Neo4j MENTIONS
expansion of stages 2-3 runs at retrieve time, reaching the graph store
through graph_rag.get_graph_store(), so a down Neo4j degrades to the plain
vector results instead of failing at engine build.
"""
import asyncio
import logging
from dataclasses import dataclass

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import EntityNode, Triplet
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app import graph_rag
from app.vector_rag import FILTER_SERVICE_KEY, build_index

logger = logging.getLogger("hybrid_rag")


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
                graph_rag.get_graph_store(),
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