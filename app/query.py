"""
THE RAG pipeline — LlamaIndex is the single retrieval interface.

There is exactly one pipeline here: the hand-written retriever and any
parallel A/B pipeline have been removed. Three retrieval branches live inside
this single LlamaIndex flow, selected by `retrieve_mode`:
  vector = the Qdrant collection (default), built by app/vector_rag.py;
  graph  = PropertyGraphIndex over Neo4j (retriever built by app/graph_rag.py,
           imported lazily so this module never needs Neo4j at import time);
  hybrid = the Qdrant chunks of the vector branch first, then a Neo4j
           expansion appends each chunk's entities and semantic triplets to
           its text — also built by app/hybrid_rag.py behind the same lazy
           import; Neo4j errors degrade to the plain vector results.
All branches return chunk TextNodes with the same provenance metadata and
share one synthesis path (same LLM, same SearchResult currency), so a
vector-vs-graph A/B never leaves this module.

  LLM        = same local llama.cpp endpoint, through llama-index-llms-openai-like.
"""
import logging
from dataclasses import dataclass
from typing import Literal

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from app.llm import SYSTEM_INSTRUCTION, get_chat_model
from app.vector_rag import FILTER_SERVICE_KEY, build_index

logger = logging.getLogger("query")


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


def build_llm() -> LLM:
    """The project-wide LLM client from app/llm.py, with the shared answer-policy
    prompt baked in as system_prompt."""
    return get_chat_model(temperature=0.2, system_prompt=SYSTEM_INSTRUCTION)


def build_query_engine(
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
    retrieve_mode: Literal["vector", "graph", "hybrid"] = "vector",
) -> RetrieverQueryEngine | BaseQueryEngine:
    """RetrieverQueryEngine: compact synthesis puts several small chunks into one prompt.

    vector: Qdrant-backed index (service_filter and collection_name apply).
    graph:  PropertyGraphIndex retriever from app/graph_rag.py; service_filter is
    ignored (no per-service filter in this MVP) and the default text-QA prompt
    is kept.
    hybrid: vector retrieval first (service_filter and collection_name apply),
    then Neo4j expansion appends graph facts to the chunk text.
    """
    if retrieve_mode == "graph":
        # Imported here, not at module top: graph retrieval needs Neo4j, the
        # vector path must not.
        from app import graph_rag

        return RetrieverQueryEngine.from_args(
            retriever=graph_rag.build_graph_retriever(top_k=top_k),
            llm=build_llm(),
            response_mode=ResponseMode.COMPACT,
        )
    if retrieve_mode == "hybrid":
        # Same lazy import: the Neo4j expansion in the hybrid branch must not
        # make a Neo4j-free vector-only startup fail at import time.
        from app import hybrid_rag

        return RetrieverQueryEngine.from_args(
            retriever=hybrid_rag.build_hybrid_retriever(
                top_k=top_k,
                service_filter=service_filter,
                collection_name=collection_name,
            ),
            llm=build_llm(),
            response_mode=ResponseMode.COMPACT,
        )
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
    """Map LlamaIndex nodes into the SearchResult shape — one common currency
    for /chat, context formatting and any future A/B."""
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
    retrieve_mode: Literal["vector", "graph", "hybrid"] = "vector",
) -> tuple[str, list[SearchResult]]:
    """
    Library-first query: retriever + ResponseSynthesizer + shared LLM client, all LlamaIndex.

    retrieve_mode selects the retrieval branch: "vector" (Qdrant), "graph"
    (PropertyGraphIndex/Neo4j) or "hybrid" (vector chunks first, Neo4j graph
    facts appended to the chunk text); synthesis and this fallback are shared.
    When the local LLM is offline, retrieval still answers: the citation-formatted
    raw context is returned instead.
    """
    engine = build_query_engine(
        top_k=top_k,
        service_filter=service_filter,
        collection_name=collection_name,
        retrieve_mode=retrieve_mode,
    )
    try:
        response = await engine.aquery(question)
        sources = _nodes_to_results(list(response.source_nodes))
        return str(response), sources
    except Exception as exc:
        logger.info(f"LlamaIndex LLM call failed ({exc}); falling back to retrieved context.")
        nodes = await engine.retriever.aretrieve(question)
        sources = _nodes_to_results(list(nodes))
        context = build_context(sources)
        reply = (
            "*(Note: Local LLM was unreachable; LlamaIndex retrieval shown directly)*\n\n"
            + context
        )
        return reply, sources