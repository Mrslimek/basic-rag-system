"""
LlamaIndex implementation of the RAG pipeline, running in parallel to the
hand-written one (app/rag_service.py) for A/B comparison.

Same infrastructure, different machinery:
  Postgres     = source of truth for chunk text (unchanged, worker-owned).
  Qdrant       = same server, SEPARATE collection (default `llama_doc_chunks`),
                 because QdrantVectorStore writes its own payload shape
                 (text + nested `metadata.*`), unlike our hand-made points.
  Embeddings   = the SAME fastembed e5-small service via a BaseEmbedding adapter.
                 Metadata is excluded from the embedded text on purpose, so the
                 vectors are byte-identical to the hand-written pipeline and the
                 two retrievers' scores are directly comparable.
  LLM          = same local llama.cpp endpoint, through llama-index-llms-openai-like.

Run the index build after corpus changes:
    uv run python -m app.llama_rag sync
"""
import asyncio
import logging
import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, MetadataInfo, VectorStoreInfo
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qm
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.embeddings import embedding_service
from app.models import Service, ServiceDocumentationChunk
from app.rag_service import SearchResult, rag_service
from app.vectorstore import get_vectorstore

logger = logging.getLogger("llama_rag")
settings = get_settings()

# Metadata attached to every indexed node. QdrantVectorStore (this version)
# flattens node.metadata into top-level payload keys and its filter converter
# uses MetadataFilter.key verbatim — so the filter/payload-index key here is
# plain "service_name" (verified against the live collection payload).
METADATA_KEYS = ("service_name", "service_id", "doc_file_path", "chunk_index")
METADATA_TYPES = {"service_name": "str", "service_id": "str", "doc_file_path": "str", "chunk_index": "num"}
FILTER_SERVICE_KEY = "service_name"

SYSTEM_INSTRUCTION = (
    "You are an expert microservices architecture assistant. "
    "Answer the user's question using ONLY the provided documentation context below. "
    "Always cite the microservice name and endpoint paths accurately. "
    "If the answer cannot be found in the context, explicitly state that."
)


class E5FastEmbed(BaseEmbedding):
    """
    Adapter: our async fastembed service, wrapped as a LlamaIndex embedding
    provider — the standard "plug your own provider into the framework" pattern.
    Guarantees the same vectors (incl. e5 asymmetric prefixes 'query: '/
    'passage: ') as the hand-written path, so A/B scores are comparable.
    """

    def _get_text_embedding(self, text: str) -> list[float]:
        # Sync API is only usable outside a running loop; our paths are all async.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(embedding_service.embed_passage(text))
        raise RuntimeError("E5FastEmbed sync API cannot be used inside a running event loop")

    def _get_query_embedding(self, query: str) -> list[float]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(embedding_service.embed_query(query))
        raise RuntimeError("E5FastEmbed sync API cannot be used inside a running event loop")

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await embedding_service.embed_passage(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Batch: one fastembed inference call for many nodes (used by IngestionPipeline).
        return await embedding_service.embed_passages(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await embedding_service.embed_query(query)


_llama_embedding: E5FastEmbed | None = None


def get_llama_embedding() -> E5FastEmbed:
    global _llama_embedding
    if _llama_embedding is None:
        _llama_embedding = E5FastEmbed()
    return _llama_embedding


_sync_client: QdrantClient | None = None


def get_sync_client() -> QdrantClient:
    """Sync client for DDL ops (create/delete collection). The async client is
    shared with app.vectorstore — one Qdrant server, two logical connections."""
    global _sync_client
    if _sync_client is None:
        _sync_client = QdrantClient(url=settings.QDRANT_URL, timeout=30.0)
    return _sync_client


def ensure_collection(collection_name: str) -> None:
    """Idempotent DDL mirroring app/vectorstore.ensure_collection: cosine over
    unit-normalized vectors + payload index for the filterable metadata key."""
    client = get_sync_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=qm.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name, FILTER_SERVICE_KEY, field_schema=qm.PayloadSchemaType.KEYWORD
        )
        logger.info(f"Created Qdrant collection '{collection_name}' for LlamaIndex")


def build_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    name = collection_name or settings.LLAMA_QDRANT_COLLECTION
    ensure_collection(name)
    return QdrantVectorStore(
        collection_name=name,
        client=get_sync_client(),
        aclient=get_vectorstore().client,
        text_key="text",
        metadata_config=VectorStoreInfo(
            metadata_info=[
                MetadataInfo(name=k, type=METADATA_TYPES[k], description=f"provenance metadata: {k}")
                for k in METADATA_KEYS
            ],
            content_info="Markdown chunk of a microservice specification (endpoints, payloads, behavior)",
        ),
        payload_indexes=[{"field_name": FILTER_SERVICE_KEY, "field_schema": qm.PayloadSchemaType.KEYWORD}],
    )


def build_index(collection_name: str | None = None) -> VectorStoreIndex:
    """Index over the existing collection — no local document store: the index is
    a derived view, Qdrant + Postgres own the data."""
    store = build_vector_store(collection_name)
    return VectorStoreIndex.from_vector_store(store, embed_model=get_llama_embedding())


def build_llm() -> OpenAILike:
    """Same llama.cpp endpoint as langchain's ChatOpenAI, LlamaIndex's client."""
    return OpenAILike(
        model=settings.LLM_MODEL,
        api_base=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        temperature=0.2,
        max_retries=1,
        timeout=60.0,
        is_chat_model=True,
        system_prompt=SYSTEM_INSTRUCTION,
    )


async def sync_from_postgres() -> dict[str, int]:
    """
    Full re-index of the LlamaIndex collection from Postgres chunks.

    Why full rebuild instead of incremental upsert: worker resyncs replace chunk
    rows with NEW uuids, so the old node ids would dangle — same orphan problem
    we solved with delete_by_service in app/worker.py, solved here by rebuild.
    Node id = chunk UUID keeps the cross-store join key identical to the
    hand-written points, so the two collections are row-for-row comparable.
    """
    name = settings.LLAMA_QDRANT_COLLECTION
    async with get_db_context() as session:
        rows = (
            await session.execute(
                select(
                    ServiceDocumentationChunk.id,
                    ServiceDocumentationChunk.content,
                    ServiceDocumentationChunk.chunk_index,
                    Service.name,
                    Service.doc_file_path,
                    Service.id.label("service_id"),
                )
                .join(Service, Service.id == ServiceDocumentationChunk.service_id)
                .order_by(Service.name, ServiceDocumentationChunk.chunk_index)
            )
        ).all()

    nodes = [
        TextNode(
            id_=str(row.id),
            text=row.content,
            metadata={
                "service_name": row.name,
                "service_id": str(row.service_id),
                "doc_file_path": row.doc_file_path,
                "chunk_index": row.chunk_index,
            },
            # Embed the raw paragraph only: identical input -> identical vector as
            # the hand-written pipeline (metadata stays visible to the LLM).
            excluded_embed_metadata_keys=list(METADATA_KEYS),
        )
        for row in rows
    ]

    get_sync_client().delete_collection(name)
    ensure_collection(name)
    store = build_vector_store(name)
    pipeline = IngestionPipeline(transformations=[get_llama_embedding()], vector_store=store)
    await pipeline.arun(nodes=nodes)

    total = get_sync_client().count(name, exact=True).count
    logger.info(f"LlamaIndex collection rebuilt: {total} points from {len(rows)} chunks")
    return {"postgres_rows": len(rows), "qdrant_points": total}


def build_query_engine(
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
):
    """RetrieverQueryEngine: compact synthesis puts several small chunks into one prompt."""
    index = build_index(collection_name)
    filters = (
        MetadataFilters(filters=[MetadataFilter(key=FILTER_SERVICE_KEY, value=service_filter)])
        if service_filter
        else None
    )
    return index.as_query_engine(
        similarity_top_k=top_k,
        llm=build_llm(),
        filters=filters,
        response_mode="compact",
    )


def _nodes_to_results(nodes: list[NodeWithScore]) -> list[SearchResult]:
    """Map LlamaIndex nodes back into OUR SearchResult shape — so /chat, context
    formatting and the A/B comparison use one common currency between pipelines."""
    return [
        SearchResult(
            service_name=n.node.metadata.get("service_name", "?"),
            doc_file_path=n.node.metadata.get("doc_file_path", "?"),
            chunk_index=int(n.node.metadata.get("chunk_index", -1)),
            content=n.node.get_content(),
            similarity_score=float(n.score or 0.0),
        )
        for n in nodes
    ]


async def aquery(
    question: str,
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
) -> tuple[str, list[SearchResult]]:
    """
    Library-first query: retriever + ResponseSynthesizer + OpenAILike, all LlamaIndex.

    Same graceful fallback as the hand-written endpoint when the local LLM is
    offline: retrieval still answers, the raw context is returned instead.
    """
    engine = build_query_engine(top_k=top_k, service_filter=service_filter, collection_name=collection_name)
    try:
        response = await engine.aquery(question)
        sources = _nodes_to_results(list(response.source_nodes))
        return str(response), sources
    except Exception as exc:
        logger.info(f"LlamaIndex LLM call failed ({exc}); falling back to retrieved context.")
        nodes = await engine.retriever.aretrieve(question)
        sources = _nodes_to_results(list(nodes))
        context = rag_service.build_context(sources)
        reply = (
            "*(Note: Local LLM was unreachable; LlamaIndex retrieval shown directly)*\n\n"
            + context
        )
        return reply, sources


async def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = await sync_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.llama_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli())
