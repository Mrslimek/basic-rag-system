"""
THE RAG pipeline — LlamaIndex is the single retrieval interface.

There is exactly one pipeline here: the hand-written retriever and any
parallel A/B pipeline have been removed. Three retrieval branches live inside
this single LlamaIndex flow, selected by `retrieve_mode`:
  vector = the Qdrant collection (default);
  graph  = PropertyGraphIndex over Neo4j (retriever built by app/graph_rag.py,
           imported lazily so this module never needs Neo4j at import time);
  hybrid = the Qdrant chunks of the vector branch first, then a Neo4j
           expansion appends each chunk's entities and semantic triplets to
           its text — also built by app/graph_rag.py behind the same lazy
           import; Neo4j errors degrade to the plain vector results.
All branches return chunk TextNodes with the same provenance metadata and
share one synthesis path (same LLM, same SearchResult currency), so a
vector-vs-graph A/B never leaves this module.

  Postgres   = source of truth for chunk text (unchanged, worker-owned).
  Qdrant     = ONE collection (default `doc_chunks`). QdrantVectorStore writes
               its own payload shape (node text + metadata flattened into
               top-level payload keys) — the hand-made points the ingest
               worker used to create are superseded by `sync` below.
  Embeddings = the process-wide e5-small model from app.embeddings
               (get_embedding_model) — the same ONNX weights, prefixes and
               normalization as always, so vectors stay interchangeable with
               existing points. Metadata is excluded from the embedded text on
               purpose: identical input -> identical vector.
  LLM        = same local llama.cpp endpoint, through llama-index-llms-openai-like.

Run the index build after corpus changes:
    uv run python -m app.llama_rag sync
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.base.base_query_engine import BaseQueryEngine
from qdrant_client import AsyncQdrantClient, QdrantClient, models as qm
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import get_embedding_model
from app.llm import SYSTEM_INSTRUCTION, get_chat_model
from app.models import Service, ServiceDocumentationChunk

logger = logging.getLogger("llama_rag")
settings = get_settings()

# Metadata attached to every indexed node. QdrantVectorStore (this version)
# flattens node.metadata into top-level payload keys and its filter converter
# uses MetadataFilter.key verbatim — so the filter/payload-index key here is
# plain "service_name" (verified against the live collection payload).
METADATA_KEYS = ("service_name", "service_id", "doc_file_path", "chunk_index")
METADATA_TYPES = {"service_name": "str", "service_id": "str", "doc_file_path": "str", "chunk_index": "num"}
FILTER_SERVICE_KEY = "service_name"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved documentation chunk with source service provenance — the
    common currency between retrieval, /chat answering and any future A/B."""
    service_name: str
    doc_file_path: str
    chunk_index: int
    content: str
    similarity_score: float


def build_context(search_results: list[SearchResult]) -> str:
    """Citation-friendly context for LLM prompts: every block carries its
    provenance so the model cannot blend endpoints across services."""
    if not search_results:
        return "No relevant documentation found."
    blocks = [
        f"### [Source {i}] Microservice: {res.service_name} "
        f"(File: {res.doc_file_path}, Chunk #{res.chunk_index}, Score: {res.similarity_score:.4f})\n"
        f"{res.content}"
        for i, res in enumerate(search_results, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def get_llama_embedding() -> BaseEmbedding:
    """The process-wide embedding model, owned by app.embeddings."""
    return get_embedding_model()


_sync_client: QdrantClient | None = None


def get_sync_client() -> QdrantClient:
    """Sync client for DDL ops (create/delete collection)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = QdrantClient(url=settings.QDRANT_URL, timeout=30)
    return _sync_client


_async_client: AsyncQdrantClient | None = None


def get_async_client() -> AsyncQdrantClient:
    """Async client for point ops — llama_rag owns its Qdrant connections."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncQdrantClient(url=settings.QDRANT_URL, timeout=30)
    return _async_client


def ensure_collection(collection_name: str) -> None:
    """Idempotent DDL: cosine over unit-normalized vectors + payload index for
    the filterable metadata key."""
    client = get_sync_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=qm.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name, FILTER_SERVICE_KEY, field_schema=qm.PayloadSchemaType.KEYWORD
        )
        logger.info(f"Created Qdrant collection '{collection_name}' for LlamaIndex")


def build_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    name = collection_name or settings.QDRANT_COLLECTION
    ensure_collection(name)
    return QdrantVectorStore(
        collection_name=name,
        client=get_sync_client(),
        aclient=get_async_client(),
        text_key="text",
        metadata_config=VectorStoreInfo(
            metadata_info=[
                MetadataInfo(name=k, type=METADATA_TYPES[k], description=f"provenance metadata: {k}")
                for k in METADATA_KEYS
            ],
            content_info="Markdown chunk of a microservice specification (endpoints, payloads, behavior)",
        ),
        payload_indexes=[{"field_name": FILTER_SERVICE_KEY, "field_schema": qm.PayloadSchemaType.KEYWORD}],
    )


def build_index(collection_name: str | None = None) -> VectorStoreIndex:
    """Index over the existing collection — no local document store: the index is
    a derived view, Qdrant + Postgres own the data."""
    store = build_vector_store(collection_name)
    return VectorStoreIndex.from_vector_store(store, embed_model=get_llama_embedding())


def build_llm() -> LLM:
    """The project-wide LLM client from app/llm.py, with the shared answer-policy
    prompt baked in as system_prompt."""
    return get_chat_model(temperature=0.2, system_prompt=SYSTEM_INSTRUCTION)


async def sync_from_postgres() -> dict[str, int]:
    """
    Full rebuild of the LlamaIndex index into the Qdrant collection.

    Why full rebuild instead of incremental upsert: worker resyncs replace chunk
    rows with NEW uuids, so the old node ids would dangle — same orphan problem
    we solved with delete_by_service in app/worker.py, solved here by rebuild.
    Node id = chunk UUID keeps the cross-store join key stable: a Qdrant point
    id resolves to exactly one Postgres chunk row.
    """
    name = settings.QDRANT_COLLECTION
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
            # Embed the raw paragraph only: identical input -> identical vector as
            # the hand-written pipeline (metadata stays visible to the LLM).
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        )
        for row in rows
    ]

    get_sync_client().delete_collection(name)
    ensure_collection(name)
    store = build_vector_store(name)
    pipeline = IngestionPipeline(transformations=[get_llama_embedding()], vector_store=store)
    await pipeline.arun(nodes=nodes)

    total = get_sync_client().count(name, exact=True).count
    logger.info(f"LlamaIndex collection rebuilt: {total} points from {len(rows)} chunks")
    return {"postgres_rows": len(rows), "qdrant_points": total}


def build_query_engine(
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
    retrieve_mode: Literal["vector", "graph", "hybrid"] = "vector",
) -> RetrieverQueryEngine | BaseQueryEngine:
    """RetrieverQueryEngine: compact synthesis puts several small chunks into one prompt.

    vector: Qdrant-backed index (service_filter and collection_name apply).
    graph:  PropertyGraphIndex retriever from app/graph_rag.py; service_filter is
    ignored (no per-service filter in this MVP) and the default text-QA prompt
    is kept.
    hybrid: vector retrieval first (service_filter and collection_name apply),
    then Neo4j expansion appends graph facts to the chunk text.
    """
    if retrieve_mode == "graph":
        # Imported here, not at module top: graph retrieval needs Neo4j, the
        # vector path must not.
        from app import graph_rag

        return RetrieverQueryEngine.from_args(
            retriever=graph_rag.build_graph_retriever(top_k=top_k),
            llm=build_llm(),
            response_mode=ResponseMode.COMPACT,
        )
    if retrieve_mode == "hybrid":
        # Same lazy import: the Neo4j expansion in the hybrid branch must not
        # make a Neo4j-free vector-only startup fail at import time.
        from app import graph_rag

        return RetrieverQueryEngine.from_args(
            retriever=graph_rag.build_hybrid_retriever(
                top_k=top_k,
                service_filter=service_filter,
                collection_name=collection_name,
            ),
            llm=build_llm(),
            response_mode=ResponseMode.COMPACT,
        )
    index = build_index(collection_name)
    filters = (
        MetadataFilters(filters=[MetadataFilter(key=FILTER_SERVICE_KEY, value=service_filter)])
        if service_filter
        else None
    )
    return index.as_query_engine(
        similarity_top_k=top_k,
        llm=build_llm(),
        filters=filters,
        response_mode="compact",
    )


def _nodes_to_results(nodes: list[NodeWithScore]) -> list[SearchResult]:
    """Map LlamaIndex nodes into the SearchResult shape — one common currency
    for /chat, context formatting and any future A/B."""
    return [
        SearchResult(
            service_name=n.node.metadata.get("service_name", "?"),
            doc_file_path=n.node.metadata.get("doc_file_path", "?"),
            chunk_index=int(n.node.metadata.get("chunk_index", -1)),
            content=n.node.get_content(),
            similarity_score=float(n.score or 0.0),
        )
        for n in nodes
    ]


async def aquery(
    question: str,
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
    retrieve_mode: Literal["vector", "graph", "hybrid"] = "vector",
) -> tuple[str, list[SearchResult]]:
    """
    Library-first query: retriever + ResponseSynthesizer + shared LLM client, all LlamaIndex.

    retrieve_mode selects the retrieval branch: "vector" (Qdrant), "graph"
    (PropertyGraphIndex/Neo4j) or "hybrid" (vector chunks first, Neo4j graph
    facts appended to the chunk text); synthesis and this fallback are shared.
    When the local LLM is offline, retrieval still answers: the citation-formatted
    raw context is returned instead.
    """
    engine = build_query_engine(
        top_k=top_k,
        service_filter=service_filter,
        collection_name=collection_name,
        retrieve_mode=retrieve_mode,
    )
    try:
        response = await engine.aquery(question)
        sources = _nodes_to_results(list(response.source_nodes))
        return str(response), sources
    except Exception as exc:
        logger.info(f"LlamaIndex LLM call failed ({exc}); falling back to retrieved context.")
        nodes = await engine.retriever.aretrieve(question)
        sources = _nodes_to_results(list(nodes))
        context = build_context(sources)
        reply = (
            "*(Note: Local LLM was unreachable; LlamaIndex retrieval shown directly)*\n\n"
            + context
        )
        return reply, sources


async def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = await sync_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.llama_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
