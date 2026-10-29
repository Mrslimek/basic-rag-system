"""
THE DOCUMENTATION SYNC WORKER — Haystack-native ingestion.

  docs_mock/*.md  = markdown service specifications, scanned every 60 seconds
                    by a taskiq schedule (LabelScheduleSource).
  Splitter        = Haystack DocumentSplitter(split_by="function", ...) driven
                    by split_markdown_sections below: ATX-heading sections,
                    byte-for-byte identical to the v1 MarkdownNodeParser
                    segmentation on this corpus (verified 52/52 against the
                    live Postgres chunk rows).
  Postgres        = source of truth: Service + ServiceDocumentationChunk rows
                    (hash-skipped when unchanged).
  Qdrant          = derived store: one point per chunk; through the Haystack
                    store API a point's id IS the chunk UUID (payload round-
                    trip), so a Qdrant point joins to exactly one Postgres row
                    and to one Neo4j Chunk node. The raw point id Qdrant keeps
                    is the store's deterministic uuid5 hash of that UUID
                    (qdrant-haystack convert_id), which is why the raw client
                    used below must apply the same mapping. Embedding happens
                    with the shared "passage: "-prefixed document embedder
                    from app.embeddings; metadata is payload-only, never
                    embedded.
  Neo4j           = NOT touched here: graph extraction is app.graph_rag's job
                    (uv run python -m app.graph_rag sync).

Resync contract: changed/new services get NEW chunk UUIDs, so the worker
captures the old chunk ids BEFORE deleting the old rows and drops the old
points directly via AsyncQdrantClient (the embedding writer itself only
upserts by id and cannot clean orphans).
"""
import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant.converters import convert_id
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, select
from taskiq import InMemoryBroker, TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from app.config import get_settings
from app.database import get_db_context, init_db
from app.embeddings import get_document_embedder
from app.models import Service, ServiceDocumentationChunk
from app.vector_rag import build_vector_store, ensure_collection

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

# Raw async client only for point deletions: index_chunks upserts current rows
# via the Haystack store but has no delete path for the previous run's points,
# so orphan cleanup goes straight to Qdrant by raw point id (mapped through
# qdrant-haystack's convert_id, the same hash the store applies on write).
_qdrant_client: AsyncQdrantClient | None = None

# ATX headings (# ... ######) open a new section; line-joined section text
# keeps internal blank lines and strips only the newlines at its edges.
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def split_markdown_sections(markdown_text: str) -> list[str]:
    """
    Split a markdown document into one section per ATX heading.

    Reproduces the v1 MarkdownNodeParser segmentation exactly: every line
    starting with "# " ... "###### " begins a new section that runs until the
    next such line; the section text keeps internal blank lines and strips
    only leading/trailing newlines. Parity with the 52 live Postgres chunk
    rows was verified byte-for-byte (wave 4, check 2).
    """
    positions = [match.start() for match in _MARKDOWN_HEADER_RE.finditer(markdown_text)]
    if not positions:
        return [markdown_text.strip("\n")] if markdown_text.strip() else []
    positions.append(len(markdown_text))
    return [
        markdown_text[start:end].strip("\n")
        for start, end in pairwise(positions)
    ]


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
    exist. QdrantDocumentStore writes each chunk under the deterministic hash
    qdrant-haystack derives from the chunk UUID (convert_id), so the raw
    client must apply the same mapping before deleting by point id.
    """
    if not old_chunk_ids:
        return
    await get_qdrant_client().delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.PointIdsList(
            points=[convert_id(str(cid)) for cid in old_chunk_ids]
        ),
    )


def index_chunks(
    chunk_rows: list[ServiceDocumentationChunk],
    service_record: Service,
    file_path: Path,
) -> None:
    """Embed and upsert chunk rows into Qdrant via the Haystack document store.

    One Document per chunk, keyed by the chunk's own UUID, so Postgres and
    Qdrant share the same id (a cross-store FK). The embedder only consumes
    Document.content; the provenance meta is payload-only, never embedded.
    Upsert is idempotent by id (DuplicatePolicy.OVERWRITE).
    """
    documents = [
        Document(
            id=str(row.id),
            content=row.content,
            meta={
                "service_name": service_record.name,
                "service_id": str(service_record.id),
                "doc_file_path": str(file_path),
                "chunk_index": row.chunk_index,
            },
        )
        for row in chunk_rows
    ]
    embedded = get_document_embedder().run(documents=documents)["documents"]
    build_vector_store().write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)


async def sync_document_file(file_path: Path) -> bool:
    """
    Splits a single markdown document into sections via the Haystack
    DocumentSplitter, compares hash with database, and updates documentation
    chunks & embeddings if modified.

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

        # 1. Chunk the document: Haystack DocumentSplitter with the heading
        # splitting function. Each split Document starts with its own heading
        # line ("## Endpoints") followed by the section body, so heading and
        # content stay together in the stored text. Empty sections are
        # discarded exactly like v1 (node.text.strip()).
        splitter = DocumentSplitter(
            split_by="function",
            splitting_function=split_markdown_sections,
        )
        split_documents = splitter.run(documents=[Document(content=content)])[
            "documents"
        ]
        chunks = [doc.content for doc in split_documents if doc.content and doc.content.strip()]
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

        # 3. Insert chunk rows (pure text in Postgres). Graph extraction
        # happens in app/graph_rag against Neo4j, not here.
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
        # the new ones through the shared Haystack embedder/store. Embedding
        # is blocking I/O, so it runs off the event loop.
        await delete_old_vectors(old_chunk_ids)
        await asyncio.to_thread(index_chunks, chunk_rows, service_record, file_path)

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