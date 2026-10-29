"""
Hybrid expander unit tests with the Neo4j half stubbed out.

FactsExpander reaches the graph only through graph_rag.query_cypher, which
these tests monkeypatch with row-shaped fakes — no Neo4j connection, no LLM,
no vector store involved.
"""
import logging
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from haystack import Document

from app import graph_rag
from app.hybrid_rag import FactsExpander, _attach_facts


def _documents() -> list[Document]:
    return [
        Document(
            id=str(uuid.uuid4()),
            content="Billing sends notifications through SendGrid.",
            meta={
                "service_name": "zzhybrid_billing",
                "service_id": str(uuid.uuid4()),
                "doc_file_path": "/mock/billing.md",
                "chunk_index": 0,
            },
            score=0.81,
        ),
        Document(
            id=str(uuid.uuid4()),
            content="Auth validates tokens against AuthService.",
            meta={
                "service_name": "zzhybrid_auth",
                "service_id": str(uuid.uuid4()),
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            score=0.42,
        ),
    ]


def _stub_query_cypher(
    mentions: list[dict[str, Any]],
    triplets: list[dict[str, Any]],
    calls: list[str],
) -> Callable[..., list[dict[str, Any]]]:
    def stub(cql: str, **params: Any) -> list[dict[str, Any]]:
        if "<-[:MENTIONS]" in cql:
            calls.append("mentions")
            return mentions
        calls.append("triplets")
        return triplets

    return stub


def test_expander_appends_deduplicated_facts_block(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = _documents()
    c1, c2 = docs[0].id, docs[1].id
    calls: list[str] = []
    monkeypatch.setattr(
        graph_rag,
        "query_cypher",
        _stub_query_cypher(
            mentions=[
                {"chunk_id": c1, "name": "SendGrid", "type": "SERVICE"},
                {"chunk_id": c1, "name": "SMTP", "type": "PROTOCOL"},
                {"chunk_id": c2, "name": "AuthService", "type": "SERVICE"},
            ],
            triplets=[
                {"source": "SendGrid", "relation": "USES", "target": "SMTP"},
                {"source": "SendGrid", "relation": "USES", "target": "SMTP"},
                {"source": "AuthService", "relation": "USES", "target": "JWT"},
                {"source": "OtherSystem", "relation": "USES", "target": "X"},
            ],
            calls=calls,
        ),
    )

    enriched = FactsExpander().run(documents=docs)["documents"]
    assert calls == ["mentions", "triplets"]

    block1 = (
        "Graph facts:\n"
        "- Entities: SendGrid (SERVICE), SMTP (PROTOCOL)\n"
        "- SendGrid -> USES -> SMTP"
    )
    block2 = (
        "Graph facts:\n"
        "- Entities: AuthService (SERVICE)\n"
        "- AuthService -> USES -> JWT"
    )
    assert enriched[0].content == f"{docs[0].content}\n\n{block1}"
    assert enriched[1].content == f"{docs[1].content}\n\n{block2}"
    assert enriched[0].content.count("SendGrid -> USES -> SMTP") == 1
    assert "OtherSystem" not in enriched[0].content
    assert enriched[0].id == c1 and enriched[0].meta == docs[0].meta
    assert enriched[0].score == pytest.approx(0.81)


def test_expander_degrades_to_unchanged_documents_on_neo4j_error(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _documents()

    def _explode(cql: str, **params: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Neo4j is down (simulated)")

    monkeypatch.setattr(graph_rag, "query_cypher", _explode)
    caplog.set_level(logging.WARNING, logger="hybrid_rag")

    out_docs = FactsExpander().run(documents=docs)["documents"]

    assert out_docs is docs
    for original, returned in zip(docs, out_docs, strict=True):
        assert returned is original
    assert "Graph facts" not in (out_docs[0].content or "")
    warnings = [
        record
        for record in caplog.records
        if "Neo4j unavailable during hybrid expansion" in record.message
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_expander_without_graph_hits_is_warningless_pass_through(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _documents()
    calls: list[str] = []
    monkeypatch.setattr(
        graph_rag, "query_cypher", _stub_query_cypher([], [], calls)
    )

    out_docs = FactsExpander().run(documents=docs)["documents"]

    assert out_docs is docs
    assert calls == ["mentions"]  # triples hop never ran
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == "hybrid_rag"
    ]


def test_attach_facts_preserves_identity_meta_and_score() -> None:
    document = Document(
        id="chunk-99",
        content="body",
        meta={"service_name": "billing", "chunk_index": 3,
              "doc_file_path": "/m/b.md"},
        score=0.42,
    )
    enriched = _attach_facts(
        document,
        {document.id: "Graph facts:\n- Entities: Stripe (SERVICE)"},
    )
    assert enriched is not document
    assert enriched.id == document.id
    assert enriched.meta == document.meta
    assert enriched.score == pytest.approx(0.42)
    assert enriched.content == "body\n\nGraph facts:\n- Entities: Stripe (SERVICE)"

    assert _attach_facts(document, {"other-id": "Graph facts:\n- nope"}) is document