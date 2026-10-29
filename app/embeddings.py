"""
Embedding providers for the Haystack pipeline: two prefix-separated ONNX
embedders over one process-wide fastembed model (intfloat/multilingual-e5-small).

Constraints that shape this module:
  1. e5-small is NOT in fastembed's stock model registry, so it must be
     registered via `TextEmbedding.add_custom_model` before any embedder runs
     (MEAN pooling + L2 normalization = what the E5 architecture produces;
     unit-length vectors make Qdrant COSINE scores equal dot products).
  2. E5 uses asymmetric retrieval prefixes — "query: " for search queries,
     "passage: " for indexed documents. fastembed-haystack applies its
     prefix/suffix parameters itself, so the prefixes are passed at construct
     time and no subclass is needed anymore (v1's E5FastEmbed is gone).
  3. One ONNX session per role: the get_* functions below are memoized so each
     embedder is a single instance per process (the model is loaded lazily by
     fastembed-haystack on first embedding call).

Owned by:
  app.vector_rag.py (index-time and query-time embedders).
"""
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedTextEmbedder,
)

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


@lru_cache
def get_document_embedder() -> FastembedDocumentEmbedder:
    """Index-time embedder: one "passage: "-prefixed instance per process."""
    _ensure_custom_model_registered()
    return FastembedDocumentEmbedder(
        model=settings.EMBEDDING_MODEL_NAME,
        prefix=PASSAGE_PREFIX,
        batch_size=256,
        progress_bar=False,
    )


def make_text_embedder() -> FastembedTextEmbedder:
    """
    A fresh query-embedder component for pipeline construction.

    Haystack forbids adding one component instance to two pipelines
    (PipelineError), so pipelines get their own instance. This is cheap:
    the ONNX session underneath is still process-wide, cached by
    fastembed-haystack's backend factory keyed on the model config.
    """
    _ensure_custom_model_registered()
    return FastembedTextEmbedder(
        model=settings.EMBEDDING_MODEL_NAME,
        prefix=QUERY_PREFIX,
        progress_bar=False,
    )


@lru_cache
def get_text_embedder() -> FastembedTextEmbedder:
    """
    Query-time embedder singleton for direct run() calls (e.g. standalone
    embedding). For pipelines use make_text_embedder — components cannot be
    shared between pipelines.
    """
    return make_text_embedder()