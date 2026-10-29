"""
Shared pytest fixtures for the v2 smoke-test suite.

Contract rules enforced here: the live docker infrastructure is used as a
test bed, but live DATA is never mutated — Qdrant tests get throwaway
zztest_* collections (doc_chunks is untouched), the Neo4j integration is
env-gated on NEO4J_TEST_URI pointing at a temp instance (the live 7687
graph is untouched), and Postgres is read-only.
"""
import uuid
from collections.abc import Iterator

import pytest
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient

from app.config import get_settings
from app.vector_rag import _vector_stores, build_vector_store


@pytest.fixture
def test_vector_store() -> Iterator[QdrantDocumentStore]:
    """
    A throwaway zztest_* collection via build_vector_store.

    Teardown drops the collection and evicts the memoized store handle, so
    no throwaway state survives the test run.
    """
    name = f"zztest_{uuid.uuid4().hex[:8]}"
    store = build_vector_store(name)
    yield store
    QdrantClient(url=get_settings().QDRANT_URL).delete_collection(name)
    _vector_stores.pop(name, None)