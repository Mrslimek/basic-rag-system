"""
Embedding smoke tests against the real ONNX model: dimensionality, run-to-run
determinism, and the asymmetric E5 prefixes (query:/passage:).

The model files live in the local HF cache (settings.HF_CACHE_DIR); these
tests exercise the cached weights only — no downloads, no paid calls.
"""
import pytest
from haystack import Document

from app.config import get_settings
from app.embeddings import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    get_document_embedder,
    get_text_embedder,
    make_text_embedder,
)

_TEXT = "How do recurring payments work under the subscription lifecycle?"


def test_embedding_dimension_is_384() -> None:
    settings = get_settings()
    assert settings.EMBEDDING_DIMENSION == 384
    result = get_document_embedder().run(documents=[Document(content=_TEXT)])
    embedding = result["documents"][0].embedding
    assert embedding is not None
    assert len(embedding) == settings.EMBEDDING_DIMENSION


def test_text_embedder_is_deterministic() -> None:
    first = get_text_embedder().run(text=_TEXT)["embedding"]
    second = get_text_embedder().run(text=_TEXT)["embedding"]
    assert first == pytest.approx(second, abs=1e-6)


def test_passage_and_query_prefixes_produce_different_vectors() -> None:
    passage = get_document_embedder().run(
        documents=[Document(content=_TEXT)]
    )["documents"][0].embedding
    query = make_text_embedder().run(text=_TEXT)["embedding"]
    assert passage is not None and query is not None
    assert any(abs(value) > 1e-6 for value in query)
    assert passage != pytest.approx(query, abs=1e-4)


def test_embedder_instances_carry_the_e5_prefixes() -> None:
    assert get_document_embedder().prefix == PASSAGE_PREFIX == "passage: "
    assert make_text_embedder().prefix == QUERY_PREFIX == "query: "