import asyncio
from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Async embedding provider wrapping `sentence-transformers` for `intfloat/multilingual-e5-small`.
    
    Architectural notes:
    1. The E5 family of embedding models uses asymmetric retrieval prefixes:
       - Passages (stored documentation chunks) must be prefixed with: "passage: "
       - Queries (user search prompts) must be prefixed with: "query: "
       This aligns the query and document vectors into the learned contrastive subspace.
    2. Model inference is CPU/GPU intensive and synchronous. We use `asyncio.to_thread`
       to offload encoding to Python's default thread pool executor, preventing blocking
       the asyncio event loop during FastAPI requests or Taskiq background jobs.
    3. We enforce `normalize_embeddings=True` to project vectors to the unit sphere (L2 norm = 1).
       This guarantees that dot product (<u, v>) equals cosine similarity, unlocking maximum
       throughput with pgvector's `vector_ip_ops` inner-product indexing.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        # Lazy loading or eager initialization of the underlying torch/sentence-transformers model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _sync_encode(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encoding worker executing in a background thread."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Convert numpy array to standard Python list of floats for pgvector/asyncpg compatibility
        return embeddings.tolist()

    async def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """
        Generates normalized embeddings for document passages to be indexed in PostgreSQL.
        Automatically applies the mandatory 'passage: ' prefix required by E5 models.
        """
        if not passages:
            return []
        prefixed = [f"passage: {p}" for p in passages]
        return await asyncio.to_thread(self._sync_encode, prefixed)

    async def embed_passage(self, passage: str) -> list[float]:
        """Convenience method to embed a single passage."""
        results = await self.embed_passages([passage])
        return results[0]

    async def embed_query(self, query: str) -> list[float]:
        """
        Generates a normalized embedding for a search query.
        Automatically applies the mandatory 'query: ' prefix required by E5 models.
        """
        prefixed = [f"query: {query}"]
        results = await asyncio.to_thread(self._sync_encode, prefixed)
        return results[0]


# Global singleton instance for reuse across requests
embedding_service = EmbeddingService()
