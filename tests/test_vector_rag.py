"""
Vector-branch smoke tests over the live Qdrant server.

Every test writes into a throwaway zztest_* collection (fixture in
conftest.py) and never touches the live doc_chunks collection; the real e5
embedder is used on both write and query sides, and no LLM is involved.
"""
import uuid

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.embeddings import get_document_embedder
from app.vector_rag import build_retrieval_pipeline

_QUERY = "How do recurring payments work?"


def _seed_documents(suffix: str, count: int = 3) -> list[Document]:
    seeds = [
        (
            "Billing Service manages subscriptions and recurring payments via Stripe.",
            f"zzvec_{suffix}_billing",
        ),
        (
            "Auth Service issues Ed25519 JWT bearer tokens and publishes JWKS keys.",
            f"zzvec_{suffix}_auth",
        ),
        (
            "Ledger Service records double-entry accounting entries in PostgreSQL.",
            f"zzvec_{suffix}_ledger",
        ),
    ]
    return [
        Document(
            id=str(uuid.uuid4()),
            content=content,
            meta={
                "service_name": service_name,
                "service_id": str(uuid.uuid4()),
                "doc_file_path": f"/mock/{service_name}.md",
                "chunk_index": 0,
            },
        )
        for content, service_name in seeds[:count]
    ]


def _write_seeds(store: QdrantDocumentStore, documents: list[Document]) -> None:
    embedded = get_document_embedder().run(documents=documents)["documents"]
    store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)


def test_write_read_roundtrip_and_service_filter(
    test_vector_store: QdrantDocumentStore,
) -> None:
    suffix = uuid.uuid4().hex[:6]
    documents = _seed_documents(suffix)
    billing = documents[0]
    _write_seeds(test_vector_store, documents)

    assert test_vector_store.count_documents() == len(documents)

    found = test_vector_store.filter_documents(
        filters={
            "field": "meta.service_name",
            "operator": "==",
            "value": f"zzvec_{suffix}_billing",
        }
    )
    assert len(found) == 1
    assert found[0].id == billing.id
    assert found[0].content == billing.content
    assert found[0].meta["service_name"] == f"zzvec_{suffix}_billing"
    assert found[0].meta["chunk_index"] == 0


def test_overwrite_upsert_is_idempotent(
    test_vector_store: QdrantDocumentStore,
) -> None:
    documents = _seed_documents(uuid.uuid4().hex[:6], count=2)
    _write_seeds(test_vector_store, documents)
    _write_seeds(test_vector_store, documents)
    assert test_vector_store.count_documents() == 2

    updated = Document(
        id=documents[0].id,
        content="Billing Service handles refunds through the Adyen gateway.",
        meta=dict(documents[0].meta),
    )
    _write_seeds(test_vector_store, [updated])
    assert test_vector_store.count_documents() == 2
    found = test_vector_store.filter_documents(
        filters={
            "field": "meta.service_name",
            "operator": "==",
            "value": documents[0].meta["service_name"],
        }
    )
    assert len(found) == 1
    assert found[0].content == updated.content


def test_retrieval_pipeline_ranks_billing_chunk_first(
    test_vector_store: QdrantDocumentStore,
) -> None:
    suffix = uuid.uuid4().hex[:6]
    _write_seeds(test_vector_store, _seed_documents(suffix))

    pipeline = build_retrieval_pipeline(top_k=2, store=test_vector_store)
    result = pipeline.run({"text_embedder": {"text": _QUERY}})
    documents = result["retriever"]["documents"]

    assert len(documents) == 2
    assert documents[0].meta["service_name"] == f"zzvec_{suffix}_billing"
    first_score = documents[0].score
    second_score = documents[1].score
    assert first_score is not None and second_score is not None
    assert first_score >= second_score


def test_retrieval_service_filter_narrows_to_one_service(
    test_vector_store: QdrantDocumentStore,
) -> None:
    suffix = uuid.uuid4().hex[:6]
    _write_seeds(test_vector_store, _seed_documents(suffix))

    pipeline = build_retrieval_pipeline(
        top_k=3,
        service_filter=f"zzvec_{suffix}_auth",
        store=test_vector_store,
    )
    result = pipeline.run({"text_embedder": {"text": "JWKS token issuance"}})
    documents = result["retriever"]["documents"]

    assert len(documents) == 1
    assert documents[0].meta["service_name"] == f"zzvec_{suffix}_auth"