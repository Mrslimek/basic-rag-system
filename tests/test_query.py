"""
Query-branch unit tests: SearchResult/context mapping, aquery with a fake
pipeline and fake generator (fallback included), and the retrieve_mode
dispatcher — including the guarantee that the vector branch never imports
the neo4j stack.
"""
import sys
from typing import Any

import pytest
from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage

from app import query
from app.query import (
    SearchResult,
    _build_pipeline,
    _documents_to_results,
    aquery,
    build_context,
)


def _result_documents() -> list[Document]:
    return [
        Document(
            id="d1",
            content="Billing retries failed payments via POST /retry.",
            meta={
                "service_name": "billing",
                "doc_file_path": "/mock/billing.md",
                "chunk_index": 2,
            },
            score=0.81,
        ),
        Document(
            id="d2",
            content="Auth issues JWT bearer tokens.",
            meta={
                "service_name": "auth",
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            score=0.55,
        ),
    ]


def test_documents_to_results_maps_meta_and_score_with_defaults() -> None:
    results = _documents_to_results(_result_documents())
    assert results == [
        SearchResult(
            service_name="billing",
            doc_file_path="/mock/billing.md",
            chunk_index=2,
            content="Billing retries failed payments via POST /retry.",
            similarity_score=0.81,
        ),
        SearchResult(
            service_name="auth",
            doc_file_path="/mock/auth.md",
            chunk_index=0,
            content="Auth issues JWT bearer tokens.",
            similarity_score=0.55,
        ),
    ]
    bare = Document(id="d3", content=None, meta={}, score=None)
    assert _documents_to_results([bare]) == [
        SearchResult(
            service_name="?",
            doc_file_path="?",
            chunk_index=-1,
            content="",
            similarity_score=0.0,
        )
    ]


def test_build_context_citations_and_empty_placeholder() -> None:
    assert build_context([]) == "No relevant documentation found."
    context = build_context(_documents_to_results(_result_documents()[0:1]))
    assert context.startswith("### [Source 1] Microservice: billing")
    assert "(File: /mock/billing.md, Chunk #2, Score: 0.8100)" in context
    assert context.endswith("Billing retries failed payments via POST /retry.")
    two = build_context(_documents_to_results(_result_documents()))
    assert "\n\n---\n\n" in two
    assert "### [Source 2] Microservice: auth" in two


class _FakePipeline:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.seen_inputs: dict[str, Any] = {}

    def run(self, inputs: dict[str, Any]) -> dict[str, dict[str, list[Document]]]:
        self.seen_inputs = inputs
        return {"retriever": {"documents": self.documents}}


class _FakePromptBuilder:
    def __init__(self) -> None:
        self.seen: dict[str, str] = {}

    def run(self, query: str, context: str) -> dict[str, list[ChatMessage]]:
        self.seen = {"query": query, "context": context}
        return {"prompt": [ChatMessage.from_user("built prompt")]}


class _OkGenerator:
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("fake answer")]}


class _BrokenGenerator:
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        raise RuntimeError("local LLM down")


def _patch_aquery(
    monkeypatch: pytest.MonkeyPatch,
    generator: _OkGenerator | _BrokenGenerator,
) -> tuple[_FakePipeline, _FakePromptBuilder]:
    documents = _result_documents()
    pipeline = _FakePipeline(documents)
    prompt_builder = _FakePromptBuilder()
    monkeypatch.setattr(
        query, "_build_pipeline", lambda **kwargs: pipeline
    )
    monkeypatch.setattr(query, "build_chat_prompt", lambda: prompt_builder)
    monkeypatch.setattr(query, "get_chat_model", lambda: generator)
    return pipeline, prompt_builder


async def test_aquery_returns_fake_reply_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, prompt_builder = _patch_aquery(monkeypatch, _OkGenerator())

    reply, sources = await aquery(
        "How do I retry a payment?", top_k=2, retrieve_mode="vector"
    )

    assert reply == "fake answer"
    assert len(sources) == 2
    assert sources[0].service_name == "billing"
    assert pipeline.seen_inputs == {"text_embedder": {"text": "How do I retry a payment?"}}
    assert prompt_builder.seen["query"] == "How do I retry a payment?"
    assert prompt_builder.seen["context"].startswith("### [Source 1]")


async def test_aquery_falls_back_to_context_when_llm_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_aquery(monkeypatch, _BrokenGenerator())

    reply, sources = await aquery("How do I retry a payment?", top_k=2)

    assert reply.startswith(
        "*(Note: Local LLM was unreachable; retrieved context shown directly)*"
    )
    assert "### [Source 1]" in reply
    assert "Billing retries failed payments" in reply
    assert len(sources) == 2


def test_dispatcher_graph_mode_calls_graph_pipeline_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import graph_rag

    sentinel = Pipeline()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        graph_rag,
        "build_graph_pipeline",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert _build_pipeline("graph", 3, None, None) is sentinel
    assert calls == [{"top_k": 3}]


def test_dispatcher_vector_mode_does_not_import_neo4j(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_rag_module = sys.modules.get("app.graph_rag")
    neo4j_module = sys.modules.get("neo4j")
    monkeypatch.setattr(
        query, "build_retrieval_pipeline", lambda **kwargs: Pipeline()
    )
    try:
        sys.modules.pop("app.graph_rag", None)
        sys.modules.pop("neo4j", None)
        assert isinstance(_build_pipeline("vector", 4, None, None), Pipeline)
        assert "neo4j" not in sys.modules
        assert "app.graph_rag" not in sys.modules
    finally:
        if graph_rag_module is not None:
            sys.modules["app.graph_rag"] = graph_rag_module
        if neo4j_module is not None:
            sys.modules["neo4j"] = neo4j_module