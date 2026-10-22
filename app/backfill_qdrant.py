"""
One-shot data migration: pgvector `embedding` column -> Qdrant points.

Why raw SQL here: the ORM model already dropped the embedding column (this is
the legacy schema), so we read it as text and parse the vector literally.

Usage:
    uv run --extra dev python -m app.backfill_qdrant          # upload + verify
    uv run --extra dev python -m app.backfill_qdrant --drop   # then drop column/extension

The --drop phase is separate on purpose: only delete the old source of vectors
after you confirmed Qdrant holds exactly the same points (it is the backup).
"""
import asyncio
import json
import sys
import uuid

from sqlalchemy import text

from app.database import engine
from app.vectorstore import ChunkVector, get_vectorstore

BATCH_SIZE = 64


async def upload_phase() -> None:
    store = get_vectorstore()
    await store.ensure_collection()

    async with engine.begin() as conn:
        # embedding::text -> asyncpg gives us the vector's text form "[0.1,0.2,...]",
        # which json.loads parses straight into list[float].
        result = await conn.execute(
            text(
                """
                SELECT c.id, c.service_id, c.chunk_index, c.embedding::text AS emb,
                       s.name AS service_name, s.doc_file_path
                FROM service_documentation_chunks c
                JOIN services s ON s.id = c.service_id
                ORDER BY c.service_id, c.chunk_index
                """
            )
        )
        rows = result.all()

    if not rows:
        print("No chunk rows (column already migrated?). Nothing to upload.")
        return

    buffer: list[ChunkVector] = []
    for row in rows:
        buffer.append(
            ChunkVector(
                chunk_id=row.id if isinstance(row.id, uuid.UUID) else uuid.UUID(str(row.id)),
                service_id=str(row.service_id),
                service_name=row.service_name,
                doc_file_path=row.doc_file_path,
                chunk_index=row.chunk_index,
                embedding=json.loads(row.emb),
            )
        )
        if len(buffer) >= BATCH_SIZE:
            await store.upsert_chunks(buffer)
            buffer = []
    if buffer:
        await store.upsert_chunks(buffer)

    total = await store.client.count(store.collection, exact=True)
    print(f"Postgres rows: {len(rows)} | Qdrant points: total_count={total.count}")


async def drop_phase() -> None:
    async with engine.begin() as conn:
        # HNSW index disappears together with its column; extension must go last.
        await conn.execute(text(
            "ALTER TABLE service_documentation_chunks DROP COLUMN IF EXISTS embedding"
        ))
        await conn.execute(text("DROP EXTENSION IF EXISTS vector"))
    print("Dropped column 'embedding' and extension 'vector' from Postgres.")


async def main() -> None:
    await upload_phase()
    if "--drop" in sys.argv:
        await drop_phase()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
