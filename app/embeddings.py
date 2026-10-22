import asyncio
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Async embedding provider wrapping ONNX Runtime via `fastembed` for `intfloat/multilingual-e5-small`.
    
    Architectural notes:
    1. Runs via ONNX Runtime (C++ engine) with zero PyTorch dependencies.
    2. The E5 family uses asymmetric retrieval prefixes:
       - Passages: "passage: "
       - Queries: "query: "
       This aligns vectors in the contrastive embedding subspace learned by the E5 model family.
    3. Output vectors are unit-normalized (L2 norm = 1.0): cosine == dot product,
       which is what Qdrant's COSINE distance scores against (see app/vectorstore.py).
    4. Model inference runs in a thread pool via `asyncio.to_thread` to keep the FastAPI event loop unblocked.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model: TextEmbedding | None = None
        self._ensure_custom_model_registered()

    def _ensure_custom_model_registered(self) -> None:
        """Register intfloat/multilingual-e5-small in fastembed's registry if not already registered."""
        try:
            TextEmbedding.add_custom_model(
                model="intfloat/multilingual-e5-small",
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf="intfloat/multilingual-e5-small"),
                dim=384,
                model_file="onnx/model.onnx",
                additional_files=[
                    "onnx/tokenizer.json",
                    "onnx/config.json",
                    "onnx/special_tokens_map.json",
                    "onnx/tokenizer_config.json",
                ],
            )
        except ValueError:
            # Already registered in this runtime
            pass

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                self.model_name,
                local_files_only=False,
            )
        return self._model

    def _sync_encode(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encoding using fastembed ONNX runtime."""
        embeddings = self.model.embed(texts)
        return [emb.tolist() for emb in embeddings]

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
