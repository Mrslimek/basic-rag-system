import uuid

import pytest
from llama_index.core.indices.property_graph import ImplicitPathExtractor, PropertyGraphIndex
from llama_index.core.schema import TextNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.llama_rag import METADATA_KEYS


def _make_nodes(suffix: str) -> list[TextNode]:
    # Same shape as the real pipeline chunks (app.llama_rag.sync_from_postgres):
    # uuid ids, METADATA_KEYS provenance, metadata excluded from the embedded text.
    return [
        TextNode(
            id_=str(uuid.uuid4()),
            text="Billing Service manages subscriptions and recurring payments via Stripe.",
            metadata={
                "service_name": f"zzgraph_{suffix}_billing",
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
                "service_name": f"zzgraph_{suffix}_auth",
                "service_id": "2",
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        ),
    ]


def _delete_and_verify_gone(store: Neo4jPropertyGraphStore, node_ids: list[str], suffix: str) -> None:
    """Id-scoped cleanup only: legacy Entity/Alias nodes must never be touched,
    so we delete our two chunk ids and assert by-id AND by-suffix counts."""
    store.delete_llama_nodes(node_ids=list(node_ids))

    assert store.get(ids=node_ids) == []
    by_id = store.structured_query(
        "MATCH (c:Chunk) WHERE c.id IN $ids RETURN count(c) AS n",
        param_map={"ids": node_ids},
    )
    assert by_id[0]["n"] == 0
    by_name = store.structured_query(
        "MATCH (c:Chunk) WHERE c.service_name CONTAINS $suffix RETURN count(c) AS n",
        param_map={"suffix": f"zzgraph_{suffix}"},
    )
    assert by_name[0]["n"] == 0


async def test_graph_store_roundtrip_without_llm():
    """Smoke test for the Neo4j graph-store plumbing WITHOUT the LLM:
    PropertyGraphIndex + ImplicitPathExtractor (no extractor ever calls an LLM)
    -> Neo4jPropertyGraphStore insert -> id roundtrip -> structured query.
    Uses throwaway chunk ids marked with a random suffix; the graph is cleaned
    unconditionally in a finally block. `use_async=False` is required: this
    core version's async insert path calls asyncio.run() internally, which
    raises inside the pytest-asyncio loop."""
    settings = get_settings()
    suffix = uuid.uuid4().hex[:8]
    node1, node2 = _make_nodes(suffix)
    ids = [node1.id_, node2.id_]

    store = Neo4jPropertyGraphStore(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )
    try:
        index = PropertyGraphIndex.from_existing(
            store,
            embed_model=get_embedding_model(),
            kg_extractors=[ImplicitPathExtractor()],
            use_async=False,
        )
        await index.ainsert_nodes([node1, node2])

        # Raw store roundtrip (sync driver): ChunkNode text + flat properties.
        fetched = {chunk.id: chunk for chunk in store.get(ids=ids)}
        assert set(fetched) == set(ids)
        assert fetched[node1.id_].text == node1.get_content()
        assert fetched[node2.id_].text == node2.get_content()
        assert fetched[node1.id_].properties["service_name"] == f"zzgraph_{suffix}_billing"

        # Reconstructed llama nodes: metadata + text survive the JSON round trip.
        restored = {n.id_: n for n in store.get_llama_nodes(ids)}
        assert set(restored) == set(ids)
        assert restored[node1.id_].get_content() == node1.get_content()
        assert restored[node1.id_].metadata["service_name"] == f"zzgraph_{suffix}_billing"
        assert restored[node2.id_].metadata["service_name"] == f"zzgraph_{suffix}_auth"

        # Graph-level: implicit path extraction labels our chunks :Chunk.
        rows = store.structured_query(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id",
            param_map={"ids": ids},
        )
        assert {row["id"] for row in rows} == set(ids)
    finally:
        _delete_and_verify_gone(store, ids, suffix)
        store.close()


def test_graph_rag_module_exposes_retriever_builder():
    """Kept green before the feature lands: app/graph_rag is optional here."""
    graph_rag = pytest.importorskip("app.graph_rag")
    assert callable(graph_rag.build_graph_retriever)
