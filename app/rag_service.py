from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.embeddings import embedding_service, EmbeddingService
from app.models import Service, ServiceDocumentationChunk


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
    Core RAG retrieval engine:
    1. Generates query embeddings with 'query: ' prefix and unit normalization.
    2. Performs vector dot-product similarity search against `service_documentation_chunks`.
    3. Joins `services` table to fetch microservice provenance metadata.
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
        Executes vector similarity search using inner product / max_inner_product.
        
        Because vectors are normalized to L2 norm = 1:
        dot_product(u, v) == cosine_similarity(u, v) ∈ [-1, 1].
        
        In pgvector, `max_inner_product(embedding, query_vector)` computes negative inner product
        `<#>` for order by ascending. To return a human-readable similarity score between -1 and 1,
        we compute: score = -1 * (embedding.max_inner_product(query_vector)).
        """
        query_vector = await self.embeddings.embed_query(query)

        # Build distance expression using pgvector's max_inner_product (<#> operator)
        distance_expr = ServiceDocumentationChunk.embedding.max_inner_product(query_vector)

        stmt = (
            select(
                Service.name.label("service_name"),
                Service.doc_file_path.label("doc_file_path"),
                ServiceDocumentationChunk.chunk_index.label("chunk_index"),
                ServiceDocumentationChunk.content.label("content"),
                (-1 * distance_expr).label("similarity_score"),
            )
            .join(Service, Service.id == ServiceDocumentationChunk.service_id)
            .order_by(distance_expr.asc())
            .limit(top_k)
        )

        if service_filter:
            stmt = stmt.where(Service.name == service_filter)

        result = await session.execute(stmt)
        rows = result.all()

        return [
            SearchResult(
                service_name=row.service_name,
                doc_file_path=row.doc_file_path,
                chunk_index=row.chunk_index,
                content=row.content,
                similarity_score=float(row.similarity_score),
            )
            for row in rows
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
