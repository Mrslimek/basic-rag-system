"""
Worker tests: pure splitter/hash/delete-id units plus one read-only
invariant against the live sandbox corpus.

The invariant test reads the 8 real docs_mock files and the live Postgres
chunk rows and asserts the v2 splitter reproduces them byte-for-byte — the
52-row v2 chunking contract. Only SELECTs: no
Postgres row, Qdrant point or Neo4j node is ever mutated here.
"""
import hashlib
import uuid

import pytest
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.document_stores.qdrant.converters import convert_id
from qdrant_client import models
from sqlalchemy import select

from app import worker
from app.config import get_settings
from app.database import get_db_context
from app.models import Service, ServiceDocumentationChunk
from app.worker import compute_file_hash, split_markdown_sections


def test_split_without_heading_returns_whole_text() -> None:
    assert split_markdown_sections("plain line\nsecond line") == [
        "plain line\nsecond line"
    ]


def test_split_empty_input_yields_no_chunks() -> None:
    assert split_markdown_sections("") == []
    assert split_markdown_sections(" \n\n ") == []


def test_split_sections_keep_internal_blank_lines() -> None:
    markdown = "# Overview\nline one\n\n\nline two\n\n## Endpoints\nPOST /retry\n"
    assert split_markdown_sections(markdown) == [
        "# Overview\nline one\n\n\nline two",
        "## Endpoints\nPOST /retry",
    ]


def test_split_drops_preamble_and_respects_atx_boundaries() -> None:
    assert split_markdown_sections("intro text\n# A\nbody\n## B\nbody2") == [
        "# A\nbody",
        "## B\nbody2",
    ]
    # "#" without a following space is not an ATX heading.
    assert split_markdown_sections("#not-a-heading\n# A\nbody") == ["# A\nbody"]


def test_document_splitter_component_runs_the_splitting_function() -> None:
    markdown = "# A\nbody\n## B\nmore"
    splitter = DocumentSplitter(
        split_by="function",
        splitting_function=split_markdown_sections,
    )
    documents = splitter.run(documents=[Document(content=markdown)])["documents"]
    assert [document.content for document in documents] == [
        "# A\nbody",
        "## B\nmore",
    ]


async def test_delete_old_vectors_maps_ids_through_convert_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, models.PointIdsList]] = []

        async def delete(
            self, collection_name: str, points_selector: models.PointIdsList
        ) -> None:
            self.calls.append((collection_name, points_selector))

    fake = RecordingClient()
    monkeypatch.setattr(worker, "get_qdrant_client", lambda: fake)
    old_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

    await worker.delete_old_vectors(old_ids)

    assert len(fake.calls) == 1
    collection_name, selector = fake.calls[0]
    assert collection_name == get_settings().QDRANT_COLLECTION
    assert selector.points == [convert_id(str(cid)) for cid in old_ids]


async def test_delete_old_vectors_is_a_noop_on_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, models.PointIdsList]] = []

        async def delete(
            self, collection_name: str, points_selector: models.PointIdsList
        ) -> None:
            self.calls.append((collection_name, points_selector))

    fake = RecordingClient()
    monkeypatch.setattr(worker, "get_qdrant_client", lambda: fake)

    await worker.delete_old_vectors([])

    assert fake.calls == []


def test_compute_file_hash_is_deterministic_and_sha256() -> None:
    content = "wave five test content\n"
    assert compute_file_hash(content) == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    assert compute_file_hash("") == hashlib.sha256(b"").hexdigest()
    assert compute_file_hash(content) != compute_file_hash(content + "x")


async def test_live_split_invariant_matches_postgres_chunks_byte_for_byte() -> None:
    settings = get_settings()
    splitter = DocumentSplitter(
        split_by="function",
        splitting_function=split_markdown_sections,
    )
    total = 0
    async with get_db_context() as session:
        services = (
            await session.execute(select(Service).order_by(Service.name))
        ).scalars().all()
        for service in services:
            path = settings.DOCS_DIR / f"{service.name}_service.md"
            assert path.is_file(), f"missing docs file for {service.name}"
            markdown = path.read_text(encoding="utf-8")
            documents = splitter.run(
                documents=[Document(content=markdown)]
            )["documents"]
            expected = [
                document.content
                for document in documents
                if document.content and document.content.strip()
            ]
            rows = (
                await session.execute(
                    select(ServiceDocumentationChunk)
                    .where(ServiceDocumentationChunk.service_id == service.id)
                    .order_by(ServiceDocumentationChunk.chunk_index)
                )
            ).scalars().all()
            actual = [row.content for row in rows]
            assert actual == expected, f"chunk mismatch for {service.name}"
            total += len(actual)
    assert total == 52