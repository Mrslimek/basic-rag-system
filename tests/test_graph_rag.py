import uuid
from typing import cast, get_args

import pytest
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    ChunkNode,
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    PropertyGraphIndex,
)
from llama_index.core.schema import TextNode, TransformComponent
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.graph_rag import (
    ENTITIES,
    EXTRACT_PROMPT_TMPL,
    RELATIONS,
    VALIDATION,
    CanonicalizingKGExtractor,
    build_kg_extractors,
    normalize_entity_name,
)
from app.vector_rag import METADATA_KEYS


def _make_nodes(suffix: str) -> list[TextNode]:
    # Same shape as the real pipeline chunks (app.vector_rag.sync_from_postgres):
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
        # The neo4j store builds ChunkNode objects (text + properties) but its stub
        # widens get() to LabelledNode, which has no .text.
        fetched = {chunk.id: cast(ChunkNode, chunk) for chunk in store.get(ids=ids)}
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


def test_normalize_entity_name_variants_collapse():
    assert normalize_entity_name("Payment Gateway") == "payment gateway"
    assert normalize_entity_name("payment gateway") == "payment gateway"
    assert normalize_entity_name("Payment-Gateway") == "payment gateway"
    assert normalize_entity_name("payment_gateway") == "payment gateway"
    assert normalize_entity_name("PAYMENT  GATEWAY") == "payment gateway"


def test_normalize_entity_name_hyphens_and_underscores():
    assert normalize_entity_name("Payment Gateway Service") == "payment gateway service"
    assert normalize_entity_name("payment-gateway-service") == "payment gateway service"
    assert normalize_entity_name("payment_gateway_service") == "payment gateway service"
    assert normalize_entity_name("  payment--gateway_ service\n") == "payment gateway service"


def test_normalize_entity_name_splits_camel_case():
    assert normalize_entity_name("PaymentGatewayService") == "payment gateway service"
    assert normalize_entity_name("AuthService") == "auth service"
    assert normalize_entity_name("PaymentsServiceStream") == "payments service stream"
    assert normalize_entity_name("UserPreferencesService") == "user preferences service"
    # CamelCase with hyphens: both rules apply, old cases stay green.
    assert normalize_entity_name("Payment-GatewayService") == "payment gateway service"


def test_normalize_entity_name_keeps_topic_names_verbatim():
    # Dotted names are topic identifiers: only strip + lowercase, no
    # underscore collapsing and no camelCase splitting.
    assert (
        normalize_entity_name("billing.invoice.payment_failed")
        == "billing.invoice.payment_failed"
    )
    assert normalize_entity_name("analytics.live.metrics") == "analytics.live.metrics"
    assert normalize_entity_name("  orders.created.v1 ") == "orders.created.v1"
    # Dot present => topic branch, camelCase is NOT split.
    assert normalize_entity_name("PaymentGateWay.Service") == "paymentgateway.service"


def test_extraction_schema_has_subscribes_to_and_authenticates_via():
    relations = set(get_args(RELATIONS))
    assert {
        "USES",
        "PROVIDES",
        "PUBLISHES_TO",
        "SUBSCRIBES_TO",
        "CALLS",
        "STORES_IN",
        "AUTHENTICATES_VIA",
    } <= relations
    assert "QUEUE" in set(get_args(ENTITIES))
    assert ("SERVICE", "SUBSCRIBES_TO", "QUEUE") in VALIDATION
    assert ("SERVICE", "AUTHENTICATES_VIA", "SERVICE") in VALIDATION


class _StubKGExtractor(TransformComponent):
    def __call__(self, nodes, show_progress=False, **kwargs):
        source = EntityNode(name="PaymentGatewayService", label="SERVICE")
        target = EntityNode(name="payment-gateway-service", label="SERVICE")
        relation = Relation(label="CALLS", source_id=source.id, target_id=target.id)
        for node in nodes:
            node.metadata[KG_NODES_KEY] = [source, target]
            node.metadata[KG_RELATIONS_KEY] = [relation]
        return nodes

    async def acall(self, nodes, show_progress=False, **kwargs):
        return self(nodes, show_progress=show_progress, **kwargs)


def test_canonicalizing_extractor_normalizes_entities_and_endpoints():
    wrapper = CanonicalizingKGExtractor(_StubKGExtractor())
    chunk = TextNode(id_="chunk-1", text="Billing calls the payment gateway.")
    processed = wrapper([chunk])

    entities = processed[0].metadata[KG_NODES_KEY]
    relations = processed[0].metadata[KG_RELATIONS_KEY]
    assert {entity.name for entity in entities} == {"payment gateway service"}
    assert all(entity.id == "payment gateway service" for entity in entities)
    assert relations[0].source_id == "payment gateway service"
    assert relations[0].target_id == "payment gateway service"


async def test_canonicalizing_extractor_acall_normalizes_entities_and_endpoints():
    wrapper = CanonicalizingKGExtractor(_StubKGExtractor())
    chunk = TextNode(id_="chunk-2", text="Billing calls the payment gateway.")
    processed = await wrapper.acall([chunk])

    entities = processed[0].metadata[KG_NODES_KEY]
    relations = processed[0].metadata[KG_RELATIONS_KEY]
    assert {entity.name for entity in entities} == {"payment gateway service"}
    assert all(entity.id == "payment gateway service" for entity in entities)
    assert relations[0].source_id == "payment gateway service"
    assert relations[0].target_id == "payment gateway service"


def test_extractor_llm_uses_settings_extract_max_tokens():
    extractors = build_kg_extractors()
    canonicalizing, implicit = extractors[0], extractors[1]
    assert isinstance(canonicalizing, CanonicalizingKGExtractor)
    assert isinstance(implicit, ImplicitPathExtractor)
    assert canonicalizing._extractor.llm.max_tokens == get_settings().extract_max_tokens
    assert canonicalizing._extractor.max_triplets_per_chunk == 8


def test_extract_prompt_defines_all_seven_relations():
    assert "{text}" in EXTRACT_PROMPT_TMPL
    assert "{max_triplets_per_chunk}" in EXTRACT_PROMPT_TMPL
    for relation in (
        "USES",
        "CALLS",
        "PROVIDES",
        "PUBLISHES_TO",
        "SUBSCRIBES_TO",
        "STORES_IN",
        "AUTHENTICATES_VIA",
    ):
        assert relation in EXTRACT_PROMPT_TMPL
    assert "never model scopes as QUEUE" in EXTRACT_PROMPT_TMPL
    assert "contain dots" in EXTRACT_PROMPT_TMPL
    assert "not invent facts" in EXTRACT_PROMPT_TMPL.lower()
