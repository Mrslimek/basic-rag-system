"""
THE VECTOR branch of the single LlamaIndex pipeline — the Qdrant-rebuild half.

  Postgres   = source of truth for chunk text (unchanged, worker-owned).
  Qdrant     = ONE collection (default `doc_chunks`), owned by this module.
               QdrantVectorStore writes its own payload shape (node text +
               metadata flattened into top-level payload keys) — the hand-made
               points the ingest worker used to create are superseded by
               `sync` below.
  Chunks     = metadata shape {service_name, service_id, doc_file_path,
               chunk_index} — METADATA_KEYS/TYPES below, flattened into
               top-level payload keys and excluded from the embedded text on
               purpose: identical input -> identical vector.
  Embeddings = the process-wide e5-small model from app.embeddings
               (get_embedding_model) — the same ONNX weights, prefixes and
               normalization as always, so vectors stay interchangeable with
               existing points.

Run the full index rebuild after corpus changes:
    uv run python -m app.vector_rag sync
"""
import asyncio
import logging
import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient, models as qm
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import get_embedding_model
from app.models import Service, ServiceDocumentationChunk

logger = logging.getLogger("vector_rag")
settings = get_settings()

# Metadata attached to every indexed node. QdrantVectorStore (this version)
# flattens node.metadata into top-level payload keys and its filter converter
# uses MetadataFilter.key verbatim — so the filter/payload-index key here is
# plain "service_name" (verified against the live collection payload).
METADATA_KEYS = ("service_name", "service_id", "doc_file_path", "chunk_index")
METADATA_TYPES = {"service_name": "str", "service_id": "str", "doc_file_path": "str", "chunk_index": "num"}
FILTER_SERVICE_KEY = "service_name"


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
    """Async client for point ops — vector_rag owns its Qdrant connections."""
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


async def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = await sync_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.vector_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())