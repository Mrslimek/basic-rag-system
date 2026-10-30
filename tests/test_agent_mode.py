"""
Agent-mode tests, all no-LLM: the retrieve_mode="agent" dispatcher branch,
the pydantic-ai tool loop driven by TestModel/FunctionModel, the
deterministic source-mapping rule (read_chunk calls + search_docs returns,
deduped, capped at MAX_SOURCES), the fallback NOTE, and the tool bodies
against fake seams (fake pipeline store, fake embedder/graph retriever/
query_cypher, fake DB session). Qdrant/Neo4j/Postgres are not required; the
one live-stack check is env-gated like the Neo4j integration tests.
"""
import importlib
import os
import sys
import uuid
from typing import Any, Self

import pytest
from haystack import Document
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from app import agent_rag, query
from app.query import SearchResult


def _pop_agent_stack() -> dict[str, Any]:
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "pydantic_ai" or name.startswith("pydantic_ai.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    return saved


def _restore_agent_stack(saved: dict[str, Any]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


def _agent_stack_loaded() -> list[str]:
    return [
        name
        for name in sys.modules
        if name == "pydantic_ai" or name.startswith("pydantic_ai.")
    ]


def _agent_documents() -> list[Document]:
    return [
        Document(
            id="11111111-1111-1111-1111-111111111111",
            content="Billing retries failed payments via POST /retry.",
            meta={
                "service_name": "billing",
                "doc_file_path": "/mock/billing.md",
                "chunk_index": 2,
            },
            score=0.81,
        ),
        Document(
            id="22222222-2222-2222-2222-222222222222",
            content="Auth issues JWT bearer tokens.",
            meta={
                "service_name": "auth",
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            score=0.55,
        ),
    ]


class _FakePipeline:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def run(self, inputs: dict[str, Any]) -> dict[str, dict[str, list[Document]]]:
        self.queries.append(inputs["text_embedder"]["text"])
        return {"retriever": {"documents": self.documents}}


class _FakeEmbedder:
    def run(self, text: str) -> dict[str, list[float]]:
        return {"embedding": [0.1, 0.2]}


class _FakeGraphRetriever:
    def __init__(self, driver: Any, top_k: int = 4) -> None:
        self.driver = driver
        self.top_k = top_k

    def run(self, query_text: str, query_embedding: list[float]) -> dict[str, list[Document]]:
        return {
            "documents": [
                Document(
                    id="33333333-3333-3333-3333-333333333333",
                    content="The Notification Service validates notification:write tokens.",
                    meta={
                        "service_name": "notification",
                        "chunk_index": 1,
                        "graph_entities": ["Notification Service", "Auth Service"],
                    },
                    score=0.9,
                )
            ]
        }


class _BrokenGraphRetriever:
    def __init__(self, driver: Any, top_k: int = 4) -> None:
        pass

    def run(self, query_text: str, query_embedding: list[float]) -> dict[str, list[Document]]:
        raise RuntimeError("bolt handshake failed")


def _patch_graph_seam(
    monkeypatch: pytest.MonkeyPatch,
    retriever: Any = _FakeGraphRetriever,
    triplets: list[dict[str, str]] | None = None,
) -> None:
    from app import graph_rag

    monkeypatch.setattr("app.embeddings.get_text_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(graph_rag, "get_driver", lambda: object())
    monkeypatch.setattr(graph_rag, "EntityGraphRetriever", retriever)
    monkeypatch.setattr(
        graph_rag, "query_cypher", lambda query, **params: triplets or []
    )


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def execute(self, query: Any) -> _FakeResult:
        return _FakeResult(self._rows)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _db_row(
    chunk_id: str, content: str = "Auth requires notification:write JWTs."
) -> dict[str, Any]:
    return {
        "id": uuid.UUID(chunk_id),
        "content": content,
        "chunk_index": 1,
        "name": "notification",
        "doc_file_path": "/mock/notification.md",
    }


# --- import-time laziness -----------------------------------------------------


def test_agent_module_import_does_not_pull_agent_stack() -> None:
    saved = _pop_agent_stack()
    try:
        # Reload re-executes the module body in place: the module object (and
        # the app package attribute) keeps its identity, so no import-system
        # state diverges after the test.
        importlib.reload(agent_rag)
        assert not _agent_stack_loaded()
    finally:
        _restore_agent_stack(saved)


def test_vector_dispatch_does_not_import_agent_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from haystack import Pipeline

    monkeypatch.setattr(query, "build_retrieval_pipeline", lambda **kwargs: Pipeline())
    saved = _pop_agent_stack()
    try:
        assert isinstance(query._build_pipeline("vector", 4, None, None), Pipeline)
        assert not _agent_stack_loaded()
    finally:
        _restore_agent_stack(saved)


# --- dispatcher branch ---------------------------------------------------------


async def test_aquery_agent_mode_delegates_to_run_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_agent(
        question: str, **kwargs: Any
    ) -> tuple[str, list[SearchResult]]:
        seen["args"] = (question, kwargs)
        return (
            "agent answer",
            [
                SearchResult(
                    service_name="billing",
                    doc_file_path="/mock/billing.md",
                    chunk_index=2,
                    content="Billing content",
                    similarity_score=0.9,
                )
            ],
        )

    monkeypatch.setattr(agent_rag, "run_agent", fake_run_agent)

    reply, sources = await query.aquery(
        "How does billing retry?",
        top_k=2,
        service_filter="billing",
        collection_name=None,
        retrieve_mode="agent",
    )

    assert reply == "agent answer"
    assert sources[0].service_name == "billing"
    assert seen["args"] == (
        "How does billing retry?",
        {
            "top_k": 2,
            "service_filter": "billing",
            "collection_name": None,
        },
    )


# --- tool bodies over fake seams ------------------------------------------------


def test_search_docs_text_formats_and_registers() -> None:
    registry: dict[str, dict[str, Any]] = {}

    output = agent_rag._search_docs_text(
        "retry", pipeline=_FakePipeline(_agent_documents()), registry=registry
    )

    lines = output.splitlines()
    assert lines[0].startswith(
        "chunk_id=11111111-1111-1111-1111-111111111111"
        " | service=billing | chunk_index=2 | score=0.8100 | "
    )
    assert "Billing retries failed payments" in lines[0]
    assert lines[1].startswith(
        "chunk_id=22222222-2222-2222-2222-222222222222"
        " | service=auth | chunk_index=0 | score=0.5500 | "
    )
    assert set(registry) == {document.id for document in _agent_documents()}
    assert registry["11111111-1111-1111-1111-111111111111"]["similarity_score"] == 0.81


def test_search_docs_text_without_hits() -> None:
    output = agent_rag._search_docs_text(
        "retry", pipeline=_FakePipeline([]), registry={}
    )
    assert output == "No matching documentation chunks found."


def test_graph_lookup_text_returns_entities_and_triplets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_graph_seam(
        monkeypatch,
        triplets=[
            {
                "source": "Notification Service",
                "relation": "AUTHENTICATES_VIA",
                "target": "Auth Service",
            }
        ],
    )

    output = agent_rag._graph_lookup_text("notification authentication")

    assert output.startswith(
        "chunk_id=33333333-3333-3333-3333-333333333333"
        " | service=notification | chunk_index=1 | "
    )
    assert "entities=[Notification Service, Auth Service]" in output
    assert "Relations:" in output
    assert "Notification Service -AUTHENTICATES_VIA-> Auth Service" in output


def test_graph_lookup_tool_text_degrades_on_neo4j_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_graph_seam(monkeypatch, retriever=_BrokenGraphRetriever)

    assert agent_rag._graph_lookup_tool_text("any topic") == (
        "graph unavailable: RuntimeError"
    )


async def test_fetch_chunk_row_returns_joined_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_id = "44444444-4444-4444-4444-444444444444"
    monkeypatch.setattr(
        agent_rag,
        "_chunk_db_session",
        lambda: _FakeSession([_db_row(chunk_id)]),
    )

    row = await agent_rag._fetch_chunk_row(chunk_id)

    assert row is not None
    assert row["content"] == "Auth requires notification:write JWTs."
    assert row["name"] == "notification"


async def test_fetch_chunk_row_missing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_rag, "_chunk_db_session", lambda: _FakeSession([]))

    assert await agent_rag._fetch_chunk_row(str(uuid.uuid4())) is None
    assert await agent_rag._fetch_chunk_row("not-a-uuid") is None


def test_read_chunk_text_registers_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_id = "44444444-4444-4444-4444-444444444444"
    registry: dict[str, dict[str, Any]] = {}

    async def fake_fetch(chunk_id: str) -> dict[str, Any] | None:
        return _db_row(chunk_id)

    monkeypatch.setattr(agent_rag, "_fetch_chunk_row", fake_fetch)

    output = agent_rag._read_chunk_text(chunk_id, registry)

    assert output == "Auth requires notification:write JWTs."
    assert registry[chunk_id]["service_name"] == "notification"
    assert registry[chunk_id]["similarity_score"] == 0.0


def test_read_chunk_text_missing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(chunk_id: str) -> dict[str, Any] | None:
        return None

    monkeypatch.setattr(agent_rag, "_fetch_chunk_row", fake_fetch)

    assert agent_rag._read_chunk_text("nope", {}) == "chunk nope not found"


# --- agent loop with TestModel (no LLM) ------------------------------------------


def test_agent_loop_completes_with_testmodel(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models.test import TestModel

    _patch_graph_seam(monkeypatch)
    registry: dict[str, dict[str, Any]] = {}
    deps = agent_rag._AgentRunDeps(
        pipeline=_FakePipeline(_agent_documents()), registry=registry
    )
    agent = agent_rag._build_agent(TestModel(call_tools="all"), deps)

    result = agent.run_sync("How does notification authenticate?", deps=deps)

    tool_names = {
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    assert tool_names == {"search_docs", "graph_lookup", "read_chunk"}
    # search_docs ran against the fake pipeline and registered real provenance
    assert {doc.id for doc in _agent_documents()} <= set(registry)
    # read_chunk hit the not-found path (TestModel args are not real ids)
    read_returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_chunk"
    ]
    assert read_returns == ["chunk a not found"]


# --- deterministic source mapping with FunctionModel (no LLM) ---------------------


def _scripted_model(tools_first_turn: list[Any]) -> Any:
    """FunctionModel function: one scripted tool-call turn, then a final text."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo

    def model_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        if not tools_first_turn:
            return ModelResponse(parts=[TextPart(content="final answer")])
        last = messages[-1]
        if any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content="final answer")])
        return ModelResponse(parts=tools_first_turn)

    return model_function


def _run_scripted_agent(
    tools_first_turn: list[Any], documents: list[Document]
) -> tuple[str, list[SearchResult], dict[str, dict[str, Any]]]:
    from pydantic_ai.models.function import FunctionModel

    registry: dict[str, dict[str, Any]] = {}
    deps = agent_rag._AgentRunDeps(
        pipeline=_FakePipeline(documents), registry=registry
    )
    agent = agent_rag._build_agent(
        FunctionModel(_scripted_model(tools_first_turn)), deps
    )
    result = agent.run_sync("question", deps=deps)
    return (
        result.output,
        agent_rag._collect_sources(result.all_messages(), registry),
        registry,
    )


def test_source_mapping_prefers_read_then_search_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ToolCallPart

    first_id = "11111111-1111-1111-1111-111111111111"
    second_id = "22222222-2222-2222-2222-222222222222"

    async def fake_fetch(chunk_id: str) -> dict[str, Any] | None:
        return _db_row(chunk_id) if chunk_id in (first_id, second_id) else None

    monkeypatch.setattr(agent_rag, "_fetch_chunk_row", fake_fetch)

    _, sources, registry = _run_scripted_agent(
        [
            ToolCallPart(
                tool_name="read_chunk", args={"chunk_id": second_id}, tool_call_id="c1"
            ),
            ToolCallPart(
                tool_name="search_docs", args={"query": "jwt"}, tool_call_id="c2"
            ),
        ],
        _agent_documents(),
    )

    # rule (a): read_chunk call order first (second_id), then rule (b): parsed
    # search_docs lines (first_id, second_id) — second_id deduped.
    assert [source.service_name for source in sources] == ["notification", "billing"]
    assert [source.similarity_score for source in sources] == [0.0, 0.81]
    assert registry  # provenance came from the executed tools


def test_source_mapping_caps_at_max_sources() -> None:
    from pydantic_ai.messages import ToolCallPart

    documents = [
        Document(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk-{i}")),
            content=f"chunk {i}",
            meta={
                "service_name": f"service-{i}",
                "doc_file_path": f"/mock/{i}.md",
                "chunk_index": i,
            },
            score=0.5,
        )
        for i in range(7)
    ]

    _, sources, _ = _run_scripted_agent(
        [
            ToolCallPart(
                tool_name="search_docs", args={"query": "x"}, tool_call_id="c1"
            )
        ],
        documents,
    )

    assert len(sources) == agent_rag.MAX_SOURCES == 5
    assert [source.service_name for source in sources] == [
        f"service-{i}" for i in range(5)
    ]


def test_source_mapping_empty_when_no_tools_called() -> None:
    output, sources, _ = _run_scripted_agent([], _agent_documents())

    assert output == "final answer"
    assert sources == []


# --- run_agent fallback ------------------------------------------------------------


def test_run_agent_falls_back_when_model_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_model() -> Any:
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(agent_rag, "build_agent_model", broken_model)

    reply, sources = agent_rag.run_agent("How does notification authenticate?")

    assert reply == agent_rag.AGENT_FALLBACK_NOTE
    assert sources == []


# --- gated live-stack check (read-only, off by default) ----------------------------


def _live_stack(func: Any) -> Any:
    """Env gate for the read-only live-stack smoke, in the spirit of the
    NEO4J_TEST_URI gating in test_graph_rag."""
    return pytest.mark.skipif(
        not os.environ.get("RAG_AGENT_LIVE_TEST"),
        reason="RAG_AGENT_LIVE_TEST not set: live-stack agent smoke skipped",
    )(func)


@_live_stack
def test_live_stack_tools_read_only() -> None:
    registry: dict[str, dict[str, Any]] = {}
    from app.vector_rag import build_retrieval_pipeline, build_vector_store

    pipeline = build_retrieval_pipeline(
        top_k=agent_rag.DEFAULT_AGENT_TOP_K, store=build_vector_store()
    )
    output = agent_rag._search_docs_text(
        "jwt signing algorithm", pipeline=pipeline, registry=registry
    )
    assert output.startswith("chunk_id=")
    first_id = next(iter(registry))
    assert agent_rag._read_chunk_text(first_id, registry)
    assert agent_rag._graph_lookup_tool_text("auth service jwt")
