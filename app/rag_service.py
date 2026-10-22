from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.embeddings import embedding_service, EmbeddingService
from app.models import Service, ServiceDocumentationChunk
from app.vectorstore import get_vectorstore


@dataclass(frozen=True)
class SearchResult:
    """Represents a retrieved documentation chunk with source service provenance."""
    service_name: str
    doc_file_path: str
    chunk_index: int
    content: str
    similarity_score: float


class RAGService:
    """
    Core RAG retrieval engine (two-store architecture):
    1. Generates query embeddings with 'query: ' prefix and unit normalization.
    2. ANN search in Qdrant: returns chunk UUIDs + cosine scores (+ payload filter
       by service_name instead of a SQL JOIN WHERE).
    3. Joins back to Postgres by id to fetch authoritative text & provenance.
    4. Formats context with clear provenance boundaries for the downstream LLM.
    """

    def __init__(self, embedding_provider: EmbeddingService = embedding_service) -> None:
        self.embeddings: EmbeddingService = embedding_provider

    async def retrieve_chunks(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 4,
        service_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Vector recall happens in Qdrant, not in Postgres anymore.

        Because e5-small vectors are L2-normalized, Qdrant's COSINE score equals
        the old pgvector inner-product score: similarity ∈ [-1..1], higher = better.

        Qdrant is the ANN index (approximate, fast); Postgres stays the source of
        truth for chunk text — we only trust rows that still exist there, so stale
        points (e.g. from an interrupted re-sync) silently drop out.
        """
        query_vector = await self.embeddings.embed_query(query)
        hits = await get_vectorstore().search(
            vector=query_vector, top_k=top_k, service_filter=service_filter
        )
        if not hits:
            return []

        # Second stage: hydrate ids -> rows with provenance. `search_vector`
        # keeps the Qdrant ranking; the SQL IN-list lookup is by primary key.
        chunk_ids = [h.chunk_id for h in hits]
        stmt = (
            select(
                ServiceDocumentationChunk.id,
                Service.name.label("service_name"),
                Service.doc_file_path.label("doc_file_path"),
                ServiceDocumentationChunk.chunk_index.label("chunk_index"),
                ServiceDocumentationChunk.content.label("content"),
            )
            .join(Service, Service.id == ServiceDocumentationChunk.service_id)
            .where(ServiceDocumentationChunk.id.in_(chunk_ids))
        )
        rows = {row.id: row for row in (await session.execute(stmt)).all()}

        return [
            SearchResult(
                service_name=rows[h.chunk_id].service_name,
                doc_file_path=rows[h.chunk_id].doc_file_path,
                chunk_index=rows[h.chunk_id].chunk_index,
                content=rows[h.chunk_id].content,
                similarity_score=h.score,
            )
            for h in hits
            if h.chunk_id in rows
        ]

    def build_context(self, search_results: list[SearchResult]) -> str:
        """
        Constructs a structured, citation-friendly context string for LLM prompts.
        Each passage includes clear microservice metadata to prevent cross-service hallucination.
        """
        if not search_results:
            return "No relevant documentation found."

        context_blocks: list[str] = []
        for i, res in enumerate(search_results, start=1):
            block = (
                f"### [Source {i}] Microservice: {res.service_name} "
                f"(File: {res.doc_file_path}, Chunk #{res.chunk_index}, Score: {res.similarity_score:.4f})\n"
                f"{res.content}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)


rag_service = RAGService()
