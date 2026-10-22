import math
import uuid
import pytest
from app.embeddings import embedding_service
from app.models import Base, Service, ServiceDocumentationChunk
from app.rag_service import RAGService, SearchResult


def test_models_structure():
    """Verify SQLAlchemy ORM model attributes and relationships."""
    service_id = uuid.uuid4()
    service = Service(
        id=service_id,
        name="test_service",
        doc_file_path="/mock/path.md",
        doc_hash="hash123",
    )
    chunk = ServiceDocumentationChunk(
        service_id=service_id,
        chunk_index=0,
        content="This is sample endpoint documentation.",
    )

    assert service.name == "test_service"
    assert chunk.chunk_index == 0
    assert ServiceDocumentationChunk.__tablename__ == "service_documentation_chunks"
    assert Service.__tablename__ == "services"


@pytest.mark.asyncio
async def test_embedding_normalization_and_dimensions():
    """
    Smoke test for embedding generation:
    1. Verifies 384 dimensions matching intfloat/multilingual-e5-small.
    2. Verifies unit vector normalization (L2 norm ≈ 1.0).
    """
    passage = "Microservice subscription billing endpoint with recurring payments"
    emb = await embedding_service.embed_passage(passage)

    assert isinstance(emb, list)
    assert len(emb) == 384

    # Verify unit length ||v|| = 1.0
    l2_norm = math.sqrt(sum(x * x for x in emb))
    assert math.isclose(l2_norm, 1.0, rel_tol=1e-4)


@pytest.mark.asyncio
async def test_semantic_similarity_search_ranking():
    """
    Verifies that normalized dot product correctly separates relevant from irrelevant passages.
    """
    billing_passage = "Billing Service manages subscriptions, Stripe payments, and invoices."
    auth_passage = "Auth Service issues Ed25519 JWT bearer tokens and JWKS public keys."

    billing_emb = await embedding_service.embed_passage(billing_passage)
    auth_emb = await embedding_service.embed_passage(auth_passage)

    # Query about payments
    query_emb = await embedding_service.embed_query("How do I charge a recurring payment subscription?")

    # Dot product calculation (equivalent to cosine similarity on unit vectors)
    billing_sim = sum(q * b for q, b in zip(query_emb, billing_emb))
    auth_sim = sum(q * a for q, a in zip(query_emb, auth_emb))

    # The billing passage should score noticeably higher than the auth passage
    assert billing_sim > auth_sim
    assert billing_sim > 0.7  # Strong similarity


def test_rag_context_builder():
    """Verifies that RAGService formats retrieved chunks with microservice provenance."""
    rag = RAGService(embedding_provider=embedding_service)
    mock_results = [
        SearchResult(
            service_name="billing",
            doc_file_path="/mock/billing.md",
            chunk_index=1,
            content="POST /api/v1/subscriptions creates a subscription.",
            similarity_score=0.8921,
        ),
        SearchResult(
            service_name="auth",
            doc_file_path="/mock/auth.md",
            chunk_index=0,
            content="Auth Service issues JWT bearer tokens.",
            similarity_score=0.7412,
        ),
    ]

    context = rag.build_context(mock_results)
    assert "Microservice: billing" in context
    assert "Microservice: auth" in context
    assert "POST /api/v1/subscriptions" in context
    assert "Score: 0.8921" in context
