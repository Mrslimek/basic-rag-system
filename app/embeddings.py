"""
Embedding provider: the single LlamaIndex `BaseEmbedding` instance for the whole
service, backed by fastembed ONNX inference of `intfloat/multilingual-e5-small`.

Constraints that shape this module:
  1. e5-small is NOT in fastembed's stock model registry, so it must be
     registered via `TextEmbedding.add_custom_model` before any instance of it
     can be constructed (MEAN pooling + L2 normalization = what the E5
     architecture produces; the vectors are unit-length, so Qdrant COSINE
     scores equal dot products).
  2. E5 uses asymmetric retrieval prefixes — "query: " for search queries,
     "passage: " for indexed documents. fastembed never applies them itself
     (its `query_embed`/`passage_embed` are plain `embed()` aliases), so this
     subclass prepends them.
  3. One ONNX model per process: `get_embedding_model()` memoizes a single
     instance, and inference is created lazily on first embedding call.
"""
from typing import Any

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from app.config import get_settings

settings = get_settings()

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def _ensure_custom_model_registered() -> None:
    """Pin e5-small into fastembed's registry; idempotent within a process."""
    try:
        TextEmbedding.add_custom_model(
            model=settings.EMBEDDING_MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=settings.EMBEDDING_MODEL_NAME),
            dim=settings.EMBEDDING_DIMENSION,
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


class E5FastEmbed(FastEmbedEmbedding):
    """
    FastEmbedEmbedding + the two constraints the stock integration does not
    know about: custom model registration and E5's asymmetric prefixes.
    """

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        # Must run BEFORE super().__init__: the parent constructs the
        # TextEmbedding there, and it fails on an unregistered model.
        _ensure_custom_model_registered()
        super().__init__(model_name=model_name or settings.EMBEDDING_MODEL_NAME, **kwargs)

    # Every public BaseEmbedding entry point (get/aget, text/query, single and
    # batch) funnels into exactly these two private leaves in the parent class;
    # prefixing here applies the prefix once on all paths, never twice.
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return super()._get_text_embeddings([f"{PASSAGE_PREFIX}{t}" for t in texts])

    def _get_query_embedding(self, query: str) -> list[float]:
        return super()._get_query_embedding(f"{QUERY_PREFIX}{query}")


_embedding_model: E5FastEmbed | None = None


def get_embedding_model() -> E5FastEmbed:
    """Process-wide singleton: one ONNX model, built on first use."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = E5FastEmbed()
    return _embedding_model
