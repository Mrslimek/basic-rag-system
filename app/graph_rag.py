"""
THE GRAPH branch of the single LlamaIndex pipeline — property-graph retrieval
parallel to the vector pipeline in app/llama_rag.py.

  Postgres  = source of truth for chunk text (unchanged, worker-owned).
  Neo4j     = PropertyGraph store written ONLY by PropertyGraphIndex: chunk
              nodes (__Node__/Chunk), entity nodes (__Node__/__Entity__),
              MENTIONS edges plus LLM-extracted typed relations. Triplet
              extraction happens inside the index via SchemaLLMPathExtractor —
              nothing hand-builds the graph.
  Retrieval = LLMSynonymRetriever (lexical graph walk) + VectorContextRetriever
              (Neo4j-native "entity" vector index, same 384-dim e5 model),
              fused by PropertyGraphIndex's retriever with text fallback.

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
from typing import Literal

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    LLMSynonymRetriever,
    PropertyGraphIndex,
    SchemaLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.core.schema import TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import get_embedding_model
from app.llama_rag import METADATA_KEYS
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
