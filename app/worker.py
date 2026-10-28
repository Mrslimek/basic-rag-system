import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, select
from taskiq import InMemoryBroker, TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from app.config import get_settings
from app.database import get_db_context, init_db
from app.embeddings import get_embedding_model
from app.models import Service, ServiceDocumentationChunk
from app.vector_rag import METADATA_KEYS, build_vector_store, ensure_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docs_sync_worker")

settings = get_settings()

# Initialize Taskiq InMemory broker (ideal for self-contained local prototype)
broker = InMemoryBroker()

# Configure scheduler with label-based periodic schedule
scheduler = TaskiqScheduler(
    broker=broker,
    sources=[LabelScheduleSource(broker)],
)

# Raw async client only for point deletions: llama-index-core 0.1x's
# IngestionPipeline.arun has no delete_ids kwarg (pipeline.py:746 signature),
# so orphan cleanup goes straight to Qdrant by point id.
_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, timeout=30)
    return _qdrant_client


def compute_file_hash(content: str) -> str:
    """Compute SHA256 hash of file content to detect document changes."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def delete_old_vectors(old_chunk_ids: list) -> None:
    """Drop the Qdrant points of previously synced chunks.

    The re-synced chunks get NEW uuids, so without this the old points would
    linger as orphans: vector hits pointing at Postgres rows that no longer
    exist. Point ids are the chunk UUIDs (str), set at indexing time.
    """
    if not old_chunk_ids:
        return
    await get_qdrant_client().delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.PointIdsList(points=[str(cid) for cid in old_chunk_ids]),
    )


async def index_chunks(
    chunk_rows: list[ServiceDocumentationChunk],
    service_record: Service,
    file_path: Path,
) -> None:
    """Embed and upsert chunk rows into Qdrant via a LlamaIndex pipeline.

    One node per chunk, keyed by the chunk's own UUID, so Postgres and Qdrant
    share the same id (a cross-store FK). Upsert is idempotent by id.
    """
    nodes = [
        TextNode(
            id_=str(row.id),
            text=row.content,
            metadata={
                "service_name": service_record.name,
                "service_id": str(service_record.id),
                "doc_file_path": str(file_path),
                "chunk_index": row.chunk_index,
            },
            # Provenance fields help retrieval filtering, not the vector itself.
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        )
        for row in chunk_rows
    ]
    pipeline = IngestionPipeline(
        transformations=[get_embedding_model()],
        vector_store=build_vector_store(),
    )
    await pipeline.arun(nodes=nodes)


async def sync_document_file(file_path: Path) -> bool:
    """
    Splits a single markdown document into sections via MarkdownNodeParser,
    compares hash with database, and updates documentation chunks &
    embeddings if modified.

    Returns:
        True if updated or inserted, False if skipped because unchanged.
    """
    service_name = file_path.stem.replace("_service", "")
    content = file_path.read_text(encoding="utf-8")
    content_hash = compute_file_hash(content)

    async with get_db_context() as session:
        # Check if service already exists
        stmt = select(Service).where(Service.name == service_name)
        result = await session.execute(stmt)
        existing_service = result.scalar_one_or_none()

        if existing_service and existing_service.doc_hash == content_hash:
            logger.info(f"Service '{service_name}' documentation unchanged (hash: {content_hash[:8]}). Skipping.")
            return False

        logger.info(f"Synchronizing documentation for microservice '{service_name}'...")

        # 1. Chunk document into markdown sections with LlamaIndex's
        # MarkdownNodeParser (pure text parsing, so the sync API is fine).
        # Verified against the parser source and docs_mock files: each node's
        # text STARTS with its own heading line ("## Endpoints") followed by
        # the section body, so heading and content stay together in stored
        # text with no prepending needed. The metadata key "header_path" holds
        # only the ANCESTOR headers (ex "/Overview/"), kept on the node for
        # retrieval-time context, not stored in the chunk row.
        # MarkdownNodeParser yields TextNode instances; the llama-index stub
        # widens the return type to BaseNode, which has no .text.
        nodes = cast(
            list[TextNode],
            MarkdownNodeParser.from_defaults().get_nodes_from_documents(
                [Document(text=content)]
            ),
        )
        chunks = [node.text for node in nodes if node.text.strip()]
        if not chunks:
            logger.warning(f"No valid markdown chunks extracted from {file_path}")
            return False

        # 2. Upsert Service record
        old_chunk_ids: list = []
        if existing_service:
            existing_service.doc_hash = content_hash
            existing_service.doc_file_path = str(file_path)
            existing_service.last_synced_at = datetime.now(UTC)
            service_record = existing_service
            # Capture old chunk ids BEFORE deleting: they double as the Qdrant
            # point ids used for orphan cleanup below.
            old_chunk_ids = list((await session.execute(
                select(ServiceDocumentationChunk.id).where(
                    ServiceDocumentationChunk.service_id == service_record.id
                )
            )).scalars().all())
            # Delete old chunks
            await session.execute(
                delete(ServiceDocumentationChunk).where(
                    ServiceDocumentationChunk.service_id == service_record.id
                )
            )
        else:
            service_record = Service(
                name=service_name,
                description=f"Documentation for {service_name} microservice",
                doc_file_path=str(file_path),
                doc_hash=content_hash,
                last_synced_at=datetime.now(UTC),
            )
            session.add(service_record)
            await session.flush()  # populate service_record.id

        # 3. Insert chunk rows (pure text in Postgres). Graph extraction happens
        # in app/graph_rag's LlamaIndex pipeline against Neo4j, not here.
        chunk_rows: list[ServiceDocumentationChunk] = []
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_row = ServiceDocumentationChunk(
                service_id=service_record.id,
                chunk_index=chunk_index,
                content=chunk_text,
            )
            session.add(chunk_row)
            chunk_rows.append(chunk_row)

        await session.flush()  # populate chunk_row.id (python-side uuid4 defaults)

        # 4. Vector side: drop the previous run's points, then embed + upsert
        # the new ones through the LlamaIndex ingestion pipeline.
        await delete_old_vectors(old_chunk_ids)
        await index_chunks(chunk_rows, service_record, file_path)

        await session.commit()
        logger.info(
            f"Successfully synced '{service_name}': {len(chunks)} chunks embedded and saved."
        )
        return True


@broker.task(
    schedule=[
        {
            "cron": "* * * * *",  # Periodic trigger every 60 seconds
        }
    ]
)
async def sync_docs_task() -> dict[str, int]:
    """
    Periodic task scanning local markdown documentation directory,
    detecting new or modified service specs, and upserting embeddings.
    """
    docs_dir = settings.DOCS_DIR
    if not docs_dir.exists():
        logger.warning(f"Docs directory '{docs_dir}' does not exist.")
        return {"scanned": 0, "updated": 0}

    markdown_files = list(docs_dir.glob("*.md"))
    updated_count = 0

    for md_file in markdown_files:
        try:
            was_updated = await sync_document_file(md_file)
            if was_updated:
                updated_count += 1
        except Exception:
            logger.exception(f"Error processing doc file {md_file}")

    logger.info(f"Sync complete. Scanned {len(markdown_files)} files, updated {updated_count}.")
    return {"scanned": len(markdown_files), "updated": updated_count}


async def run_worker() -> None:
    """Entrypoint for running the worker and scheduler together."""
    logger.info("Initializing database schemas and Qdrant collection...")
    await init_db()
    # ensure_collection is sync DDL; offload so the event loop stays free.
    await asyncio.to_thread(ensure_collection, settings.QDRANT_COLLECTION)
    logger.info("Running initial documentation sync immediately on startup...")
    await sync_docs_task()

    logger.info("Starting Taskiq worker and scheduler (polling every 60 seconds)...")
    await broker.startup()
    await scheduler.startup()
    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await scheduler.shutdown()
        await broker.shutdown()


if __name__ == "__main__":
    asyncio.run(run_worker())
