import uuid

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from app.vector_rag import (
    FILTER_SERVICE_KEY,
    METADATA_KEYS,
    build_vector_store,
    get_llama_embedding,
    get_sync_client,
)


def _make_nodes(suffix: str) -> list[TextNode]:
    # Node ids must be valid Qdrant point ids (uuid or unsigned int) — in the real
    # pipeline they are the chunk UUIDs; here fresh random ones.
    return [
        TextNode(
            id_=str(uuid.uuid4()),
            text="Billing Service manages subscriptions and recurring payments via Stripe.",
            metadata={
                "service_name": f"billing-{suffix}",
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
                "service_name": f"auth-{suffix}",
                "service_id": "2",
                "doc_file_path": "/mock/auth.md",
                "chunk_index": 0,
            },
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        ),
    ]


async def test_llama_ingestion_retrieval_and_metadata_filter():
    """Smoke test for the LlamaIndex pipeline WITHOUT the LLM:
    IngestionPipeline embeds our nodes -> Qdrant collection -> retriever ranks
    the billing node first for a payments query, and MetadataFilters restrict
    recall to a single service (the framework equivalent of service_filter).
    Uses a throwaway collection named with a random suffix."""
    suffix = uuid.uuid4().hex[:8]
    collection = f"test_llama_{suffix}"
    try:
        store = build_vector_store(collection)
        pipeline = IngestionPipeline(transformations=[get_llama_embedding()], vector_store=store)
        await pipeline.arun(nodes=_make_nodes(suffix))

        assert get_sync_client().count(collection, exact=True).count == 2

        index = VectorStoreIndex.from_vector_store(store, embed_model=get_llama_embedding())

        hits = await index.as_retriever(similarity_top_k=2).aretrieve(
            "How do recurring payments work?"
        )
        assert hits
        assert hits[0].node.metadata["service_name"] == f"billing-{suffix}"
        # Score is cosine on our shared e5 vectors -> comparable to the hand-written path.
        top_score = hits[0].score
        last_score = hits[-1].score
        assert len(hits) == 1 or (
            top_score is not None and last_score is not None and top_score > last_score
        )

        filtered = index.as_retriever(
            similarity_top_k=2,
            filters=MetadataFilters(
                filters=[MetadataFilter(key=FILTER_SERVICE_KEY, value=f"auth-{suffix}")]
            ),
        )
        hits = await filtered.aretrieve("subscription payments refund")
        assert len(hits) == 1
        assert all(h.node.metadata["service_name"] == f"auth-{suffix}" for h in hits)
    finally:
        get_sync_client().delete_collection(collection)
