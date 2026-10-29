"""
Graph-branch tests.

Units below run without any Neo4j connection (name normalization, triplet
parsing, TripletParser). The writer/retriever integration tests are env-gated:
they run only when NEO4J_TEST_URI points at a throwaway temp instance and the
module-level get_driver is monkeypatched to it, so the live 7687 graph is
never opened by the test suite.
"""
import json
import logging
import os
import uuid
from typing import Any

import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage
from neo4j import Driver, GraphDatabase

from app import graph_rag
from app.config import get_settings
from app.embeddings import make_text_embedder
from app.graph_rag import (
    MAX_TRIPLETS_PER_CHUNK,
    EntityGraphRetriever,
    KGRecord,
    Neo4jGraphWriter,
    TripletParser,
    _parse_triplet,
    _parse_triplets,
    ensure_entity_index,
    normalize_entity_name,
)

# ---------------------------------------------------------------- name units

def test_normalize_entity_name_keeps_dotted_topics_verbatim() -> None:
    assert (
        normalize_entity_name("billing.invoice.payment_failed")
        == "billing.invoice.payment_failed"
    )
    assert normalize_entity_name("  Orders.Created.V1 ") == "orders.created.v1"
    # Dot present => topic branch: camelCase is NOT split, only lowercased.
    assert normalize_entity_name("PaymentGateWay.Service") == "paymentgateway.service"


def test_normalize_entity_name_splits_camel_case() -> None:
    assert normalize_entity_name("PaymentGatewayService") == "payment gateway service"
    assert normalize_entity_name("AuthService") == "auth service"


def test_normalize_entity_name_collapses_hyphens_and_underscores() -> None:
    assert normalize_entity_name("Payment-Gateway") == "payment gateway"
    assert normalize_entity_name("payment_gateway") == "payment gateway"
    assert normalize_entity_name("PAYMENT  GATEWAY\n") == "payment gateway"
    assert (
        normalize_entity_name("payment--gateway_ service")
        == "payment gateway service"
    )


# -------------------------------------------------------------- parser units

def test_parse_triplet_normalizes_and_uppercases_types() -> None:
    record = _parse_triplet(
        {
            "head": "Billing",
            "head_type": "service",
            "relation": "uses",
            "tail": "stripe",
            "tail_type": "service",
        }
    )
    assert record == KGRecord(
        head="billing",
        head_type="SERVICE",
        relation="USES",
        tail="stripe",
        tail_type="SERVICE",
    )


def test_parse_triplets_keeps_valid_and_discards_invalid() -> None:
    payload: dict[str, Any] = {
        "triplets": [
            {
                "head": "Billing",
                "head_type": "SERVICE",
                "relation": "PUBLISHES_TO",
                "tail": "billing.payments.failed",
                "tail_type": "QUEUE",
            },
            # Invalid entity type.
            {"head": "x", "head_type": "SCOPE", "relation": "USES",
             "tail": "y", "tail_type": "SERVICE"},
            # Invalid relation.
            {"head": "x", "head_type": "SERVICE", "relation": "HAS",
             "tail": "y", "tail_type": "SERVICE"},
            # Pair outside the VALIDATION whitelist.
            {"head": "x", "head_type": "DATABASE", "relation": "USES",
             "tail": "y", "tail_type": "DATABASE"},
            # Non-dict item.
            "not-a-dict",
            # Blank head name.
            {"head": " ", "head_type": "SERVICE", "relation": "USES",
             "tail": "y", "tail_type": "SERVICE"},
        ]
    }
    assert _parse_triplets(payload) == [
        KGRecord(
            head="billing",
            head_type="SERVICE",
            relation="PUBLISHES_TO",
            tail="billing.payments.failed",
            tail_type="QUEUE",
        )
    ]


def test_parse_triplets_accepts_the_full_triplet_budget() -> None:
    payload = {
        "triplets": [
            {
                "head": f"service{i}",
                "head_type": "SERVICE",
                "relation": "CALLS",
                "tail": f"service{i + 1}",
                "tail_type": "SERVICE",
            }
            for i in range(MAX_TRIPLETS_PER_CHUNK)
        ]
    }
    assert len(_parse_triplets(payload)) == MAX_TRIPLETS_PER_CHUNK


# ------------------------------------------------------- TripletParser units

def test_triplet_parser_run_on_a_valid_reply() -> None:
    reply = ChatMessage.from_assistant(
        json.dumps(
            {
                "triplets": [
                    {
                        "head": "Billing",
                        "head_type": "SERVICE",
                        "relation": "PUBLISHES_TO",
                        "tail": "billing.events.failed",
                        "tail_type": "QUEUE",
                    }
                ]
            }
        )
    )
    records = TripletParser().run(replies=[reply])["records"]
    assert records == [
        KGRecord(
            head="billing",
            head_type="SERVICE",
            relation="PUBLISHES_TO",
            tail="billing.events.failed",
            tail_type="QUEUE",
        )
    ]


def test_triplet_parser_run_logs_warning_on_malformed_json(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="graph_rag")
    records = TripletParser().run(
        replies=[ChatMessage.from_assistant("{definitely not json")]
    )["records"]
    assert records == []
    assert any("malformed JSON" in record.message for record in caplog.records)


def test_triplet_parser_run_on_empty_or_nondict_payloads() -> None:
    parser = TripletParser()
    assert parser.run(replies=[]) == {"records": []}
    assert parser.run(replies=[ChatMessage.from_assistant("[]")]) == {"records": []}
    assert parser.run(replies=[ChatMessage.from_assistant("null")]) == {"records": []}


# ------------------------------------------------- temp-Neo4j integrations

_NEO4J_TEST_URI = os.environ.get("NEO4J_TEST_URI")
needs_temp_neo4j = pytest.mark.skipif(
    not _NEO4J_TEST_URI,
    reason="NEO4J_TEST_URI not set: temp-Neo4j integration skipped",
)


def _patch_driver_to_temp(monkeypatch: pytest.MonkeyPatch) -> Driver:
    settings = get_settings()
    uri = _NEO4J_TEST_URI
    assert uri is not None
    driver = GraphDatabase.driver(
        uri,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        max_transaction_retry_time=5.0,
    )
    monkeypatch.setattr(graph_rag, "get_driver", lambda: driver)
    ensure_entity_index(driver)
    return driver


def _test_chunks(suffix: str) -> tuple[Document, Document]:
    chunk1 = Document(
        id=str(uuid.uuid4()),
        content="Billing publishes failed payment events to a Kafka topic.",
        meta={
            "service_name": f"zzgraph_{suffix}_billing",
            "service_id": str(uuid.uuid4()),
            "doc_file_path": "/mock/billing.md",
            "chunk_index": 0,
        },
    )
    chunk2 = Document(
        id=str(uuid.uuid4()),
        content="Auth issues Ed25519 JWT bearer tokens and JWKS keys.",
        meta={
            "service_name": f"zzgraph_{suffix}_auth",
            "service_id": str(uuid.uuid4()),
            "doc_file_path": "/mock/auth.md",
            "chunk_index": 0,
        },
    )
    return chunk1, chunk2


@needs_temp_neo4j
def test_writer_roundtrip_and_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _patch_driver_to_temp(monkeypatch)
    suffix = uuid.uuid4().hex[:8]
    chunk1, chunk2 = _test_chunks(suffix)
    chunk_ids = [chunk1.id, chunk2.id]
    head = f"billing service-{suffix}"
    topic = f"billing.invoice.payment_failed-{suffix}"
    records = [
        KGRecord(
            head=head,
            head_type="SERVICE",
            relation="PUBLISHES_TO",
            tail=topic,
            tail_type="QUEUE",
        )
    ]
    try:
        writer = Neo4jGraphWriter(driver)
        assert writer.run(records=records, documents=[chunk1, chunk2]) == {"written": 2}

        rows = graph_rag.query_cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.service_name AS name",
            ids=chunk_ids,
        )
        assert {row["id"] for row in rows} == set(chunk_ids)
        assert {row["name"] for row in rows} == {
            f"zzgraph_{suffix}_billing",
            f"zzgraph_{suffix}_auth",
        }

        entities = graph_rag.query_cypher(
            "MATCH (e:Entity) WHERE e.name IN $names "
            "RETURN e.name AS name, labels(e) AS labels, e.embedding AS embedding",
            names=[head, topic],
        )
        assert {row["name"] for row in entities} == {head, topic}
        assert {label for row in entities for label in row["labels"]} >= {
            "Entity",
            "SERVICE",
            "QUEUE",
        }
        for row in entities:
            assert row["embedding"] and len(row["embedding"]) == 384

        mentions = graph_rag.query_cypher(
            "MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) WHERE c.id IN $ids "
            "RETURN count(*) AS n",
            ids=chunk_ids,
        )[0]["n"]
        assert mentions == 4  # 2 entities x 2 chunks

        relations = graph_rag.query_cypher(
            "MATCH (h:Entity {name: $head})-[r:PUBLISHES_TO]->"
            "(t:Entity {name: $tail}) RETURN count(r) AS n",
            head=head,
            tail=topic,
        )[0]["n"]
        assert relations == 1

        # Re-run must keep every test-scoped count identical (MERGE semantics).
        assert writer.run(records=records, documents=[chunk1, chunk2]) == {"written": 2}
        assert graph_rag.query_cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN count(c) AS n", ids=chunk_ids
        )[0]["n"] == 2
        assert graph_rag.query_cypher(
            "MATCH (e:Entity) WHERE e.name IN $names RETURN count(e) AS n",
            names=[head, topic],
        )[0]["n"] == 2
        assert graph_rag.query_cypher(
            "MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) WHERE c.id IN $ids "
            "RETURN count(*) AS n",
            ids=chunk_ids,
        )[0]["n"] == 4
        assert graph_rag.query_cypher(
            "MATCH (h:Entity {name: $head})-[r:PUBLISHES_TO]->"
            "(t:Entity {name: $tail}) RETURN count(r) AS n",
            head=head,
            tail=topic,
        )[0]["n"] == 1
    finally:
        graph_rag.query_cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids DETACH DELETE c", ids=chunk_ids
        )
        graph_rag.query_cypher(
            "MATCH (e:Entity) WHERE e.name CONTAINS $suffix DETACH DELETE e",
            suffix=suffix,
        )
        assert graph_rag.query_cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN count(c) AS n", ids=chunk_ids
        )[0]["n"] == 0
        assert graph_rag.query_cypher(
            "MATCH (e:Entity) WHERE e.name CONTAINS $suffix RETURN count(e) AS n",
            suffix=suffix,
        )[0]["n"] == 0
        driver.close()


@needs_temp_neo4j
def test_retriever_returns_documents_with_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _patch_driver_to_temp(monkeypatch)
    suffix = uuid.uuid4().hex[:8]
    chunk1, chunk2 = _test_chunks(suffix)
    chunk_ids = [chunk1.id, chunk2.id]
    head = f"billing service-{suffix}"
    topic = f"billing.invoice.payment_failed-{suffix}"
    records = [
        KGRecord(
            head=head,
            head_type="SERVICE",
            relation="PUBLISHES_TO",
            tail=topic,
            tail_type="QUEUE",
        )
    ]
    try:
        Neo4jGraphWriter(driver).run(records=records, documents=[chunk1, chunk2])

        query_embedding = make_text_embedder().run(
            text="billing publishes payment events"
        )["embedding"]
        retriever = EntityGraphRetriever(driver=driver, top_k=2)
        documents = retriever.run(
            query_text="How does billing publish payment events?",
            query_embedding=query_embedding,
        )["documents"]

        assert documents
        assert {document.id for document in documents} <= set(chunk_ids)
        for document in documents:
            assert document.meta["service_name"].startswith(f"zzgraph_{suffix}_")
            assert document.meta["doc_file_path"]
            assert document.score is not None and document.score > 0
    finally:
        graph_rag.query_cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids DETACH DELETE c", ids=chunk_ids
        )
        graph_rag.query_cypher(
            "MATCH (e:Entity) WHERE e.name CONTAINS $suffix DETACH DELETE e",
            suffix=suffix,
        )
        driver.close()