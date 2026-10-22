import uuid
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.database import get_db_context, init_db
from app.models import (
    EdgeMention,
    EntityAlias,
    GraphEdge,
    GraphEntity,
    Service,
    ServiceDocumentationChunk,
)


async def _upsert_edge(session, source_id, target_id, relation) -> uuid.UUID:
    """
    Insert the (source, target, relation) triple, or reuse the existing row.
    This is the idiom ingestion will use for every extracted relation.
    """
    stmt = (
        pg_insert(GraphEdge)
        .values(source_id=source_id, target_id=target_id, relation=relation,
                description="to validate tokens")
        .on_conflict_do_nothing(index_elements=["source_id", "target_id", "relation"])
        .returning(GraphEdge.id)
    )
    edge_id = (await session.execute(stmt)).scalar()
    if edge_id is None:
        # Conflict: DO NOTHING skipped the row — and RETURNING yields nothing
        # for skipped rows, so the id must be fetched separately.
        edge_id = await session.scalar(
            select(GraphEdge.id).where(
                GraphEdge.source_id == source_id,
                GraphEdge.target_id == target_id,
                GraphEdge.relation == relation,
            )
        )
    return edge_id


@pytest.mark.asyncio
async def test_edge_triple_dedup_and_mention_accumulation():
    """One fact = one edge; repeated chunks add evidence, not duplicates."""
    await init_db()
    suffix = uuid.uuid4().hex[:8]

    async with get_db_context() as session:
        service = Service(
            name=f"probe-{suffix}",
            doc_file_path=f"/mock/probe-{suffix}.md",
            doc_hash="probe",
        )
        session.add(service)
        await session.flush()

        chunk_a = ServiceDocumentationChunk(
            service_id=service.id, chunk_index=0,
            content="Billing calls Auth Service to validate tokens.",
        )
        chunk_b = ServiceDocumentationChunk(
            service_id=service.id, chunk_index=1,
            content="Billing calls Auth Service to validate tokens. (repeated)",
        )
        billing = GraphEntity(canonical_name=f"billing-{suffix}", entity_type="service")
        auth = GraphEntity(canonical_name=f"auth-{suffix}", entity_type="service")
        session.add_all([chunk_a, chunk_b, billing, auth])
        await session.flush()

        try:
            first = await _upsert_edge(session, billing.id, auth.id, "calls")
            second = await _upsert_edge(session, billing.id, auth.id, "calls")
            assert first == second

            session.add_all([
                EdgeMention(edge_id=first, chunk_id=chunk_a.id),
                EdgeMention(edge_id=first, chunk_id=chunk_b.id),
            ])
            await session.flush()

            edge_count = await session.scalar(
                select(func.count()).select_from(GraphEdge).where(GraphEdge.id == first)
            )
            mention_count = await session.scalar(
                select(func.count()).select_from(EdgeMention).where(EdgeMention.edge_id == first)
            )
            assert edge_count == 1
            assert mention_count == 2

            # Surface forms resolve to canonical nodes through the alias table.
            session.add(EntityAlias(alias="AuthService", normalized="authservice", entity_id=auth.id))
            await session.flush()
            resolved_id = await session.scalar(
                select(EntityAlias.entity_id).where(EntityAlias.normalized == "authservice")
            )
            assert resolved_id == auth.id
        finally:
            # Deleting entities cascades edges and mentions; deleting the
            # service cascades its chunks.
            await session.execute(
                delete(GraphEntity).where(GraphEntity.id.in_([billing.id, auth.id]))
            )
            await session.execute(delete(Service).where(Service.id == service.id))
            await session.commit()


@pytest.mark.asyncio
async def test_self_loop_edges_are_rejected():
    """CHECK constraint: an entity must never relate to itself."""
    await init_db()
    async with get_db_context() as session:
        solo = GraphEntity(canonical_name=f"solo-{uuid.uuid4().hex[:8]}", entity_type="service")
        session.add(solo)
        await session.flush()

        session.add(GraphEdge(source_id=solo.id, target_id=solo.id, relation="calls"))
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()  # nothing was committed; leave DB clean
