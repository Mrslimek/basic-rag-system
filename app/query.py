"""
THE RAG pipeline — Haystack is the single retrieval interface.

One synthesis path serves three retrieval branches, selected by
`retrieve_mode`:
  vector = the Qdrant collection (default): text_embedder -> retriever,
           built by app/vector_rag.py (imported at module top);
  graph  = the Neo4j v2 graph: text_embedder -> graph_retriever, built by
           app/graph_rag.py — imported lazily so this module never needs the
           neo4j driver at import time;
  hybrid = the Qdrant chunks of the vector branch first, then a Neo4j
           expansion appends each chunk's graph facts to its text — built by
           app/hybrid_rag.py behind the same lazy import; Neo4j errors
           degrade to the plain vector documents.
All branches return chunk Documents with the same provenance metadata and
score currency and share ONE synthesis path here (SearchResult ->
build_context -> ChatPromptBuilder -> OpenAIChatGenerator), so a
vector-vs-graph A/B never leaves this module.

  LLM        = the local OpenAI-protocol endpoint through app/llm.py
               (get_chat_model / build_chat_prompt).
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

from haystack import Document, Pipeline

from app.llm import build_chat_prompt, get_chat_model
from app.vector_rag import build_retrieval_pipeline, build_vector_store

logger = logging.getLogger("query")

_FINAL_COMPONENTS: dict[str, str] = {
    "vector": "retriever",
    "graph": "graph_retriever",
    "hybrid": "facts_expander",
}


@dataclass(frozen=True)
class SearchResult:
    """A retrieved documentation chunk with source service provenance — the
    common currency between retrieval, /chat answering and any future A/B."""
    service_name: str
    doc_file_path: str
    chunk_index: int
    content: str
    similarity_score: float


def build_context(search_results: list[SearchResult]) -> str:
    """Citation-friendly context for LLM prompts: every block carries its
    provenance so the model cannot blend endpoints across services."""
    if not search_results:
        return "No relevant documentation found."
    blocks = [
        f"### [Source {i}] Microservice: {res.service_name} "
        f"(File: {res.doc_file_path}, Chunk #{res.chunk_index}, Score: {res.similarity_score:.4f})\n"
        f"{res.content}"
        for i, res in enumerate(search_results, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def _build_pipeline(
    retrieve_mode: Literal["vector", "graph", "hybrid"],
    top_k: int,
    service_filter: str | None,
    collection_name: str | None,
) -> Pipeline:
    """Retrieval branch dispatcher.

    vector:  app.vector_rag.build_retrieval_pipeline; service_filter and
             collection_name apply (the store is only built when an explicit
             collection is requested; otherwise the memoized default).
    graph:   app.graph_rag.build_graph_pipeline — imported lazily, inside the
             branch body, so a Neo4j-free vector-only startup must not fail
             at import time; service_filter is ignored (no per-service filter
             in this MVP).
    hybrid:  app.hybrid_rag.build_hybrid_pipeline behind the same lazy import.
    """
    if retrieve_mode == "graph":
        from app import graph_rag

        return graph_rag.build_graph_pipeline(top_k=top_k)
    if retrieve_mode == "hybrid":
        from app import hybrid_rag

        return hybrid_rag.build_hybrid_pipeline(
            top_k=top_k,
            service_filter=service_filter,
            collection_name=collection_name,
        )
    store = build_vector_store(collection_name) if collection_name else None
    return build_retrieval_pipeline(
        top_k=top_k,
        service_filter=service_filter,
        store=store,
    )


def _documents_to_results(documents: list[Document]) -> list[SearchResult]:
    """Map Haystack Documents into the SearchResult shape — one common
    currency for /chat, build_context and any future A/B."""
    return [
        SearchResult(
            service_name=doc.meta.get("service_name", "?"),
            doc_file_path=doc.meta.get("doc_file_path", "?"),
            chunk_index=int(doc.meta.get("chunk_index", -1)),
            content=doc.content or "",
            similarity_score=float(doc.score or 0.0),
        )
        for doc in documents
    ]


async def aquery(
    question: str,
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
    retrieve_mode: Literal["vector", "graph", "hybrid"] = "vector",
) -> tuple[str, list[SearchResult]]:
    """
    Library-first query: Haystack retrieval branch + the single shared LLM synthesis.

    retrieve_mode selects the retrieval branch: "vector" (Qdrant), "graph"
    (Neo4j v2 graph) or "hybrid" (vector chunks first, Neo4j graph facts
    appended to the chunk text). Retrieval errors propagate to the caller;
    only an LLM synthesis failure falls back to the raw citation-formatted
    context, so retrieval still answers when the local LLM is offline.
    """
    pipeline = _build_pipeline(
        retrieve_mode=retrieve_mode,
        top_k=top_k,
        service_filter=service_filter,
        collection_name=collection_name,
    )
    inputs: dict[str, dict[str, Any]] = {"text_embedder": {"text": question}}
    if retrieve_mode == "graph":
        # The graph retriever also consumes the raw query text (keyword
        # scoring) alongside the embedding from the shared text_embedder.
        inputs["graph_retriever"] = {"query_text": question}

    result = await asyncio.to_thread(pipeline.run, inputs)
    documents = result[_FINAL_COMPONENTS[retrieve_mode]]["documents"]
    sources = _documents_to_results(documents)
    context = build_context(sources)

    try:
        prompt = build_chat_prompt().run(query=question, context=context)["prompt"]
        reply = get_chat_model().run(messages=prompt)["replies"][-1].text or ""
    except Exception as exc:  # noqa: BLE001 - synthesis failure must fall back to the raw retrieved context
        logger.info("LLM call failed (%s); falling back to retrieved context.", exc)
        reply = (
            "*(Note: Local LLM was unreachable; retrieved context shown directly)*\n\n"
            + context
        )
    return reply, sources