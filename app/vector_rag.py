"""
THE VECTOR branch of the Haystack pipeline — the Qdrant-rebuild-and-retrieve half.

  Postgres   = source of truth for chunk text (unchanged, worker-owned).
  Qdrant     = ONE collection (default `doc_chunks`), owned by this module.
               QdrantDocumentStore writes chunks as points whose id is the
               chunk UUID and whose payload nests provenance under "meta." —
               the filter key is therefore "meta.service_name".
  Chunks     = metadata shape {service_name, service_id, doc_file_path,
               chunk_index} — METADATA_KEYS below. Metadata is stored in the
               payload but never embedded: identical input -> identical vector.
  Embeddings = the process-wide e5-small embedders from app.embeddings
               (get_document_embedder / get_text_embedder) — same ONNX
               weights and "passage: "/"query: " prefixes as always.
  Indexing   = sync_from_postgres does a FULL rebuild, not an incremental
               upsert: the worker replaces chunk rows with NEW uuids on
               resync, so old point ids would dangle — the same orphan
               problem we solved with delete_by_service in app/worker.py,
               solved here by rebuild. Point id = chunk UUID keeps the
               cross-store join key stable: a Qdrant point id resolves to
               exactly one Postgres chunk row.

Run the full index rebuild after corpus changes:
    uv run python -m app.vector_rag sync
"""
import asyncio
import logging
import sys

from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient
from sqlalchemy import RowMapping, select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import get_document_embedder, make_text_embedder
from app.models import Service, ServiceDocumentationChunk

logger = logging.getLogger("vector_rag")
settings = get_settings()

# Metadata attached to every indexed Document. QdrantDocumentStore nests the
# meta dict under the "meta." payload namespace (verified against the qdrant
# payload built by qdrant-haystack 13.1.0), so retrieval filters must use the
# "meta.service_name" field path — FILTER_SERVICE_KEY holds the bare key.
METADATA_KEYS = ("service_name", "service_id", "doc_file_path", "chunk_index")
FILTER_SERVICE_KEY = "service_name"


def ensure_collection(index_name: str) -> QdrantDocumentStore:
    """
    Store over `index_name`, creating the collection + service_name payload
    index on first use. The QdrantDocumentStore constructor itself performs
    the idempotent DDL, so there is no separate collection-management path.
    """
    return QdrantDocumentStore(
        url=settings.QDRANT_URL,
        index=index_name,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        recreate_index=False,
        wait_result_from_api=True,
        progress_bar=False,
        payload_fields_to_index=[
            {"field_name": FILTER_SERVICE_KEY, "field_schema": "keyword"},
        ],
    )


# One collection = one store per process: stores hold client pools and are
# collection-specific, so they are memoized per index name, not per call.
_vector_stores: dict[str, QdrantDocumentStore] = {}


def build_vector_store(index_name: str | None = None) -> QdrantDocumentStore:
    """Memoized store for the default collection (or an explicit index name)."""
    name = index_name or settings.QDRANT_COLLECTION
    if name not in _vector_stores:
        _vector_stores[name] = ensure_collection(name)
    return _vector_stores[name]


def build_retrieval_pipeline(
    top_k: int = 4,
    service_filter: str | None = None,
    store: QdrantDocumentStore | None = None,
) -> Pipeline:
    """
    Retrieval pipeline: text_embedder (query prefix) -> retriever over Qdrant.

    Run contract:
        result = pipeline.run({"text_embedder": {"text": q}})
        result["retriever"]["documents"]  # list[Document] with id/score/meta

    service_filter narrows the search to one service via the legacy dict
    filter syntax ({"field": "meta.service_name", "operator": "==", ...}).
    """
    filters = None
    if service_filter:
        filters = {
            "field": f"meta.{FILTER_SERVICE_KEY}",
            "operator": "==",
            "value": service_filter,
        }

    pipeline = Pipeline()
    # make_text_embedder, not get_text_embedder: one component instance
    # cannot be added to two pipelines (PipelineError), and this factory is
    # called once per query (service_filter varies).
    pipeline.add_component("text_embedder", make_text_embedder())
    pipeline.add_component(
        "retriever",
        QdrantEmbeddingRetriever(
            document_store=store or build_vector_store(),
            top_k=top_k,
            filters=filters,
        ),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline


def _load_chunk_rows() -> list[RowMapping]:
    """
    Chunk rows joined with service provenance, ordered for a stable rebuild.

    The query runs inside a short-lived event loop: app.database exposes only
    the asyncpg async session, while this module (and the Haystack pipeline
    it feeds) is synchronous. sync_from_postgres is a CLI entry point, so no
    loop is running at call time.
    """

    async def _fetch() -> list[RowMapping]:
        async with get_db_context() as session:
            result = await session.execute(
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
            return list(result.mappings())

    return asyncio.run(_fetch())


def _build_documents(rows: list[RowMapping]) -> list[Document]:
    """Chunk rows -> Documents with the METADATA_KEYS provenance payload."""
    return [
        Document(
            id=str(row["id"]),
            content=row["content"],
            meta={
                "service_name": row["name"],
                "service_id": str(row["service_id"]),
                "doc_file_path": row["doc_file_path"],
                "chunk_index": row["chunk_index"],
            },
        )
        for row in rows
    ]


def sync_from_postgres() -> dict[str, int]:
    """
    Full rebuild of the Qdrant collection from Postgres chunks.

    Why full rebuild instead of incremental upsert: worker resyncs replace
    chunk rows with NEW uuids, so the old point ids would dangle — solved
    here by dropping and recreating the whole collection. Point id = chunk
    UUID keeps the cross-store join key stable.
    """
    rows = _load_chunk_rows()
    name = settings.QDRANT_COLLECTION

    ddl_client = QdrantClient(url=settings.QDRANT_URL)
    if ddl_client.collection_exists(name):
        ddl_client.delete_collection(name)

    # A cached store would hold a lazy client pointed at the just-deleted
    # collection; drop it so build_vector_store constructs a fresh store
    # (its constructor recreates the collection).
    _vector_stores.pop(name, None)
    store = build_vector_store(name)

    documents = _build_documents(rows)
    embedded = get_document_embedder().run(documents=documents)["documents"]
    store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)

    total = store.count_documents()
    logger.info("Haystack collection rebuilt: %d points from %d chunks", total, len(rows))
    return {"postgres_rows": len(rows), "qdrant_points": total}


def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = sync_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.vector_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()