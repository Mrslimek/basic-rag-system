import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import delete, select
from taskiq import InMemoryBroker, TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from app.chunker import chunk_markdown_paragraphs
from app.config import get_settings
from app.database import get_db_context, init_db
from app.embeddings import embedding_service
from app.models import Service, ServiceDocumentationChunk

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


def compute_file_hash(content: str) -> str:
    """Compute SHA256 hash of file content to detect document changes."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def sync_document_file(file_path: Path) -> bool:
    """
    Parses a single markdown document, compares hash with database,
    and updates documentation chunks & embeddings if modified.
    
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

        # 1. Chunk document
        chunks = chunk_markdown_paragraphs(content)
        if not chunks:
            logger.warning(f"No valid markdown chunks extracted from {file_path}")
            return False

        # 2. Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = await embedding_service.embed_passages(chunk_texts)

        # 3. Upsert Service record
        if existing_service:
            existing_service.doc_hash = content_hash
            existing_service.doc_file_path = str(file_path)
            existing_service.last_synced_at = datetime.now(timezone.utc)
            service_record = existing_service
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
                last_synced_at=datetime.now(timezone.utc),
            )
            session.add(service_record)
            await session.flush()  # populate service_record.id

        # 4. Insert new chunks with vector embeddings
        for chunk_data, emb in zip(chunks, embeddings):
            chunk_row = ServiceDocumentationChunk(
                service_id=service_record.id,
                chunk_index=chunk_data.index,
                content=chunk_data.content,
                embedding=emb,
            )
            session.add(chunk_row)

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
        except Exception as e:
            logger.error(f"Error processing doc file {md_file}: {e}", exc_info=True)

    logger.info(f"Sync complete. Scanned {len(markdown_files)} files, updated {updated_count}.")
    return {"scanned": len(markdown_files), "updated": updated_count}


async def run_worker() -> None:
    """Entrypoint for running the worker and scheduler together."""
    logger.info("Initializing database schemas and pgvector extension...")
    await init_db()
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
