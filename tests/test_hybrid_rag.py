import logging
import uuid
from typing import cast, get_args

import pytest
from llama_index.core.graph_stores.types import Relation
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    PropertyGraphIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app import graph_rag, hybrid_rag
from app.config import get_settings
from app.embeddings import get_embedding_model
from app.vector_rag import (
    METADATA_KEYS,
    build_index,
    build_vector_store,
    get_llama_embedding,
    get_sync_client,
)

_QUERY = "How do recurring payments work?"


def _make_nodes(suffix: str) -> list[TextNode]:
    # Same texts and metadata shape as the seeds in test_graph_rag.py and
    # test_vector_rag.py, so the same TextNode objects (same uuid ids) can
    # drive both the Qdrant and the Neo4j seeding.
    return [
        TextNode(
            id_=str(uuid.uuid4()),
            text="Billing Service manages subscriptions and recurring payments via Stripe.",
            metadata={
                "service_name": f"zzhybrid_{suffix}_billing",
                "service_id": "1",
                "doc_file_path": "/mock/billing.md",
                "chunk_index": 0,
            },
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        ),
        TextNode(
            id_=str(uuid.uuid4()),
            text="Auth Service issues Ed25519 JWT bearer tokens and publishes JWKS public keys.",
            metadata={
                "service_name": f"zzhybrid_{suffix}_auth",
                "service_id": "2",
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        ),
    ]


# ImplicitPathExtractor (this llama-index-core version) emits no entities at
# all — probe verified zero MENTIONS edges for any seed text — so the smoke
# test plants the exact structure the Neo4j store's own upsert_nodes writes
# for LLM-extracted entities: __Node__/__Entity__ node carrying a type label,
# triplet_source_id on the entity (delete_llama_nodes keys cleanup on it), and
# a directed MENTIONS edge out of the chunk.
_ENTITY_UPSERT_CYPHER = """
UNWIND $data AS row
MERGE (e:__Node__ {id: row.id})
SET e += apoc.map.clean(row.properties, [], [])
SET e.name = row.name, e:__Entity__
WITH e, row
CALL apoc.create.addLabels(e, [row.label]) YIELD node
WITH e, row
MERGE (c:__Node__ {id: row.properties.triplet_source_id})
MERGE (e)<-[:MENTIONS]-(c)
"""


def _seed_mentions(
    store: Neo4jPropertyGraphStore, billing_id: str, auth_id: str, suffix: str
) -> None:
    store.structured_query(
        _ENTITY_UPSERT_CYPHER,
        param_map={
            "data": [
                {
                    "id": f"Billing Service-{suffix}",
                    "name": f"Billing Service-{suffix}",
                    "label": "SERVICE",
                    "properties": {"triplet_source_id": billing_id},
                },
                {
                    "id": f"Kafka-{suffix}",
                    "name": f"Kafka-{suffix}",
                    "label": "QUEUE",
                    "properties": {"triplet_source_id": billing_id},
                },
                {
                    "id": f"JWT-{suffix}",
                    "name": f"JWT-{suffix}",
                    "label": "PROTOCOL",
                    "properties": {"triplet_source_id": auth_id},
                },
            ]
        },
    )
    store.upsert_relations(
        [
            Relation(
                source_id=f"Billing Service-{suffix}",
                target_id=f"Kafka-{suffix}",
                label="PUBLISHES_TO",
            )
        ]
    )


def _delete_and_verify_gone(store: Neo4jPropertyGraphStore, node_ids: list[str], suffix: str) -> None:
    # Id-scoped cleanup in the style of test_graph_rag.py; entities carry
    # triplet_source_id = chunk id, so delete_llama_nodes removes them too.
    store.delete_llama_nodes(node_ids=list(node_ids))

    assert store.get(ids=node_ids) == []
    by_id = store.structured_query(
        "MATCH (c:Chunk) WHERE c.id IN $ids RETURN count(c) AS n",
        param_map={"ids": node_ids},
    )
    assert by_id[0]["n"] == 0
    by_name = store.structured_query(
        "MATCH (c:Chunk) WHERE c.service_name CONTAINS $suffix RETURN count(c) AS n",
        param_map={"suffix": f"zzhybrid_{suffix}"},
    )
    assert by_name[0]["n"] == 0
    entities = store.structured_query(
        "MATCH (e:__Entity__) WHERE e.name CONTAINS $frag RETURN count(e) AS n",
        param_map={"frag": f"-{suffix}"},
    )
    assert entities[0]["n"] == 0


async def test_hybrid_appends_graph_facts_and_preserves_vector_stage():
    settings = get_settings()
    suffix = uuid.uuid4().hex[:8]
    collection = f"test_hybrid_{suffix}"
    billing, auth = _make_nodes(suffix)
    ids = [billing.id_, auth.id_]

    store = Neo4jPropertyGraphStore(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )
    try:
        vector_store = build_vector_store(collection)
        pipeline = IngestionPipeline(transformations=[get_llama_embedding()], vector_store=vector_store)
        await pipeline.arun(nodes=[billing, auth])
        assert get_sync_client().count(collection, exact=True).count == 2

        index = PropertyGraphIndex.from_existing(
            store,
            embed_model=get_embedding_model(),
            kg_extractors=[ImplicitPathExtractor()],
            use_async=False,
        )
        await index.ainsert_nodes([billing, auth])

        _seed_mentions(store, billing.id_, auth.id_, suffix)

        # The MENTIONS hop of the hybrid branch, probed live on this seed.
        probe = store.structured_query(
            "MATCH (c:Chunk{id:$id})-[:MENTIONS]->(e) RETURN labels(e) AS l, e.id AS name",
            param_map={"id": billing.id_},
        )
        assert {row["name"] for row in probe} == {f"Billing Service-{suffix}", f"Kafka-{suffix}"}
        assert any("SERVICE" in row["l"] for row in probe)

        retriever = hybrid_rag.build_hybrid_retriever(top_k=2, collection_name=collection)
        hits = await retriever.aretrieve(_QUERY)

        assert all(h.node.node_id in ids for h in hits)
        assert hits[0].node.node_id == billing.id_
        assert hits[0].node.metadata["service_name"] == f"zzhybrid_{suffix}_billing"
        assert all(h.score is not None and h.score > 0 for h in hits)

        by_id = {h.node.node_id: h for h in hits}
        billing_text = by_id[billing.id_].node.get_content()
        auth_text = by_id[auth.id_].node.get_content()

        assert "Graph facts:" in billing_text
        assert f"Billing Service-{suffix} (SERVICE)" in billing_text
        assert f"Kafka-{suffix} (QUEUE)" in billing_text
        assert f"- Billing Service-{suffix} -> PUBLISHES_TO -> Kafka-{suffix}" in billing_text
        assert "Graph facts:" in auth_text
        assert f"JWT-{suffix} (PROTOCOL)" in auth_text

        base = build_index(collection_name=collection).as_retriever(similarity_top_k=2)
        base_hits = await base.aretrieve(_QUERY)
        assert [h.node.node_id for h in hits] == [b.node.node_id for b in base_hits]
        for h, b in zip(hits, base_hits):
            assert h.score == pytest.approx(b.score, rel=1e-6)
    finally:
        get_sync_client().delete_collection(collection)
        _delete_and_verify_gone(store, ids, suffix)
        store.close()


async def test_hybrid_degrades_to_vector_results_when_neo4j_down(caplog, monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    collection = f"test_hybrid_{suffix}"
    try:
        vector_store = build_vector_store(collection)
        pipeline = IngestionPipeline(transformations=[get_llama_embedding()], vector_store=vector_store)
        await pipeline.arun(nodes=_make_nodes(suffix))

        class _DownStore:
            def structured_query(self, query, param_map=None):
                raise RuntimeError("Neo4j is down (simulated)")

        monkeypatch.setattr(graph_rag, "get_graph_store", lambda: _DownStore())
        caplog.set_level(logging.WARNING, logger="hybrid_rag")

        retriever = hybrid_rag.build_hybrid_retriever(top_k=2, collection_name=collection)
        hits = await retriever.aretrieve(_QUERY)

        base = build_index(collection_name=collection).as_retriever(similarity_top_k=2)
        base_hits = await base.aretrieve(_QUERY)
        assert [h.node.node_id for h in hits] == [b.node.node_id for b in base_hits]
        assert [h.node.get_content() for h in hits] == [b.node.get_content() for b in base_hits]
        for h, b in zip(hits, base_hits):
            assert h.score == pytest.approx(b.score, rel=1e-6)
        assert all("Graph facts" not in h.node.get_content() for h in hits)

        warnings = [r for r in caplog.records if "Neo4j unavailable during hybrid expansion" in r.message]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
    finally:
        get_sync_client().delete_collection(collection)


def test_hybrid_wiring_build_query_engine_and_chat_request(monkeypatch):
    from app.query import build_query_engine
    from main import ChatRequest

    def _no_graph() -> Neo4jPropertyGraphStore:
        raise AssertionError("hybrid build must not touch Neo4j")

    monkeypatch.setattr(graph_rag, "get_graph_store", _no_graph)
    engine = build_query_engine(top_k=2, retrieve_mode="hybrid")
    # build_query_engine's union return type exists only because as_query_engine
    # is stubbed as BaseQueryEngine; every branch returns a RetrieverQueryEngine.
    assert isinstance(cast(RetrieverQueryEngine, engine).retriever, hybrid_rag._HybridRetriever)

    assert get_args(ChatRequest.model_fields["retrieve_mode"].annotation) == (
        "vector",
        "graph",
        "hybrid",
    )