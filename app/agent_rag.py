"""
THE AGENT branch — a pydantic-ai tool loop over the live stack.

  Model      = app.config LLM_API_BASE / LLM_MODEL / LLM_API_KEY through
               pydantic-ai's OpenAIChatModel (any OpenAI-protocol endpoint is
               a base_url swap). The model must emit native tool calls: the
               local llama.cpp 1.5B build does not (see README), so agentic
               runs are meant for a tool-calling model.
  Tools      = search_docs (Qdrant retrieval via app.vector_rag factories),
               graph_lookup (Neo4j entities + triplets via app.graph_rag),
               read_chunk (full chunk text from Postgres). All read-only.
  Sources    = mapped deterministically from the tool-call transcript, never
               from the LLM's "Sources:" line: read_chunk call args first,
               then the chunk ids parsed from search_docs output lines, in
               that order, deduped, capped at MAX_SOURCES. Provenance comes
               from the same Document meta as vector mode, captured by a
               per-run registry as the tools execute.

Heavy framework imports (pydantic_ai, the haystack-backed app modules, the
neo4j driver via app.graph_rag) are lazy inside function bodies, mirroring
the graph branch's lazy import in app.query: importing app.agent_rag must
stay cheap, and the API process only pays for pydantic-ai when an agentic
query actually runs.
"""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from pydantic_ai.agent import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.query import SearchResult

logger = logging.getLogger("agent_rag")

AGENT_REQUEST_LIMIT = 12
AGENT_TOOL_CALL_LIMIT = 12
DEFAULT_AGENT_TOP_K = 4
MAX_SOURCES = 5
_AGENT_MODEL_TEMPERATURE = 0.2
_GRAPH_LOOKUP_TOP_K = 6
_SNIPPET_CHARS = 300

# Same fallback marker as the synthesis fallback in app.query — one NOTE
# prefix across every mode keeps the frontend/test contract uniform.
AGENT_FALLBACK_NOTE = (
    "*(Note: Local LLM was unreachable; retrieved context shown directly)*"
)

AGENT_SYSTEM_PROMPT = (
    "You are an expert microservices architecture assistant working over a "
    "documentation toolset. Use search_docs for direct factual lookups in the "
    "indexed documentation. Use graph_lookup for relationships, entities and "
    "multi-hop questions about services. Use read_chunk to read a promising "
    "candidate chunk in full. Answer strictly from the material the tools "
    "returned — never from prior knowledge. End your answer with one final "
    "line 'Sources: <service_name>:<chunk_index> ...' listing every chunk you "
    "used."
)


@dataclass
class _AgentRunDeps:
    """Per-run state closed over by the tools: the retrieval pipeline and the
    chunk provenance registry keyed by chunk id."""

    pipeline: Any | None
    registry: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_agent_model() -> Model:
    """The agent's chat model from app.config: any OpenAI-protocol endpoint
    reachable at LLM_API_BASE (no hardcoded keys, models or urls)."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    settings = get_settings()
    return OpenAIChatModel(
        settings.LLM_MODEL,
        provider=OpenAIProvider(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
        ),
    )


def _register_document(
    registry: dict[str, dict[str, Any]], document: Any
) -> None:
    """Capture one Qdrant Document's provenance — the same meta paths and
    defaults as app.query._documents_to_results."""
    if not document.id:
        return
    registry[str(document.id)] = {
        "service_name": document.meta.get("service_name", "?"),
        "doc_file_path": document.meta.get("doc_file_path", "?"),
        "chunk_index": int(document.meta.get("chunk_index", -1)),
        "content": document.content or "",
        "similarity_score": float(document.score or 0.0),
    }


def _search_docs_text(
    query: str,
    *,
    pipeline: Any,
    registry: dict[str, dict[str, Any]],
) -> str:
    """search_docs body: one run of the vector retrieval pipeline, formatted
    one line per document so the chunk id stays parseable from the reply."""
    result = pipeline.run({"text_embedder": {"text": query}})
    documents = result["retriever"]["documents"]
    if not documents:
        return "No matching documentation chunks found."
    lines = []
    for document in documents:
        _register_document(registry, document)
        snippet = (document.content or "").replace("\n", " ")[:_SNIPPET_CHARS]
        lines.append(
            f"chunk_id={document.id} | service={document.meta.get('service_name', '?')} "
            f"| chunk_index={int(document.meta.get('chunk_index', -1))} "
            f"| score={float(document.score or 0.0):.4f} | {snippet}"
        )
    return "\n".join(lines)


def _graph_lookup_text(topic: str) -> str:
    """graph_lookup body: entity resolution (keyword + vector KNN) over the
    Neo4j graph, then one triplets hop — the FactsExpander semantics from
    app.hybrid_rag, packaged as tool facts."""
    from app import graph_rag
    from app.embeddings import get_text_embedder
    from app.hybrid_rag import _TRIPLETS_HOP_QUERY

    embedding = get_text_embedder().run(topic)["embedding"]
    retriever = graph_rag.EntityGraphRetriever(
        driver=graph_rag.get_driver(), top_k=_GRAPH_LOOKUP_TOP_K
    )
    documents = retriever.run(query_text=topic, query_embedding=embedding)[
        "documents"
    ]
    if not documents:
        return "No entities matched this topic in the knowledge graph."
    names = sorted(
        {
            name
            for document in documents
            for name in document.meta.get("graph_entities", [])
        }
    )
    triplets = (
        graph_rag.query_cypher(_TRIPLETS_HOP_QUERY, names=names) if names else []
    )
    lines = []
    for document in documents:
        entities = ", ".join(document.meta.get("graph_entities", []))
        snippet = (document.content or "").replace("\n", " ")[:_SNIPPET_CHARS]
        lines.append(
            f"chunk_id={document.id} | service={document.meta.get('service_name', '?')} "
            f"| chunk_index={int(document.meta.get('chunk_index', -1))} "
            f"| entities=[{entities}] "
            f"| score={float(document.score or 0.0):.4f} | {snippet}"
        )
    if triplets:
        lines.append("Relations:")
        lines.extend(
            f"{t['source']} -{t['relation']}-> {t['target']}" for t in triplets
        )
    return "\n".join(lines)


def _graph_lookup_tool_text(topic: str) -> str:
    """Tool wrapper: a Neo4j failure degrades to a marker string so the agent
    loop keeps running instead of crashing the run."""
    try:
        return _graph_lookup_text(topic)
    except Exception as exc:  # noqa: BLE001 - any graph failure must degrade
        logger.info("graph_lookup failed (%s); degrading to marker.", exc)
        return f"graph unavailable: {type(exc).__name__}"


@asynccontextmanager
async def _chunk_db_session() -> AsyncGenerator[AsyncSession]:
    """One-shot Postgres session for a read_chunk call.

    The shared engine in app.database pools asyncpg connections that are bound
    to the event loop they were created on; a sync tool spins a FRESH
    asyncio.run loop per call, so a pooled connection from a previous call's
    loop fails with "attached to a different loop" (observed live on the
    second read_chunk of one run). A throwaway NullPool engine per call never
    reuses connections across loops; the extra local TCP connect is negligible
    against the tool's LLM-loop cadence.
    """
    from sqlalchemy.ext.asyncio import (
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.pool import NullPool

    settings = get_settings()
    engine = create_async_engine(
        settings.DATABASE_URL, echo=False, poolclass=NullPool
    )
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        await engine.dispose()


async def _fetch_chunk_row(chunk_id: str) -> Mapping[str, Any] | None:
    """Full chunk row joined with its service provenance — the same Postgres
    join app.vector_rag._load_chunk_rows uses, narrowed to one id."""
    from sqlalchemy import select

    from app.models import Service, ServiceDocumentationChunk

    try:
        chunk_uuid = uuid.UUID(chunk_id)
    except ValueError:
        return None
    async with _chunk_db_session() as session:
        result = await session.execute(
            select(
                ServiceDocumentationChunk.id,
                ServiceDocumentationChunk.content,
                ServiceDocumentationChunk.chunk_index,
                Service.name,
                Service.doc_file_path,
            )
            .join(Service, Service.id == ServiceDocumentationChunk.service_id)
            .where(ServiceDocumentationChunk.id == chunk_uuid)
        )
        return next(iter(result.mappings()), None)


def _read_chunk_text(
    chunk_id: str, registry: dict[str, dict[str, Any]]
) -> str:
    """read_chunk body: full chunk content from Postgres; the row's
    provenance feeds the registry (no similarity score exists for a direct
    read, so it registers as 0.0)."""
    row = asyncio.run(_fetch_chunk_row(chunk_id))
    if row is None:
        return f"chunk {chunk_id} not found"
    registry[str(row["id"])] = {
        "service_name": row["name"],
        "doc_file_path": row["doc_file_path"],
        "chunk_index": int(row["chunk_index"]),
        "content": row["content"],
        "similarity_score": 0.0,
    }
    return str(row["content"])


def _build_agent(model: Model, deps: _AgentRunDeps) -> Agent[_AgentRunDeps, str]:
    """Assemble the agent: system prompt + the three read-only tools. Factored
    out of run_agent so tests can drive the loop with TestModel/FunctionModel
    and fake seams instead of a paid model."""
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.settings import ModelSettings

    agent = Agent(
        model,
        instructions=AGENT_SYSTEM_PROMPT,
        deps_type=_AgentRunDeps,
        # Per-request timeout mirrors app.llm (300s): one hung endpoint request
        # must surface as a run failure (fallback note), not block forever.
        model_settings=ModelSettings(
            temperature=_AGENT_MODEL_TEMPERATURE, timeout=300.0
        ),
    )

    @agent.tool
    def search_docs(ctx: RunContext[_AgentRunDeps], query: str) -> str:
        """Search the microservice documentation in the vector store. Returns
        matching chunks with chunk_id, service name, chunk index and score."""
        return _search_docs_text(
            query, pipeline=ctx.deps.pipeline, registry=ctx.deps.registry
        )

    @agent.tool_plain
    def graph_lookup(topic: str) -> str:
        """Look up a topic in the knowledge graph. Returns matching entities,
        the chunks they are mentioned in and the typed relations between
        entities."""
        return _graph_lookup_tool_text(topic)

    @agent.tool
    def read_chunk(ctx: RunContext[_AgentRunDeps], chunk_id: str) -> str:
        """Read the full text of one documentation chunk from Postgres by its
        chunk_id (as printed by the other tools)."""
        return _read_chunk_text(chunk_id, ctx.deps.registry)

    return agent


def _chunk_ids_from_search_output(output: str) -> list[str]:
    """Parse chunk ids from search_docs reply lines in output order — one
    'chunk_id=<id> | ...' line per document."""
    ids = []
    for line in output.splitlines():
        if line.startswith("chunk_id="):
            ids.append(line[len("chunk_id=") :].split(" | ", 1)[0])
    return ids


def _source_ids_from_messages(
    messages: Sequence[ModelMessage],
) -> tuple[list[str], list[str]]:
    """Rule inputs of the deterministic source mapping: chunk ids from
    read_chunk tool CALLS (rule a) and from search_docs tool RETURNS (rule
    b), each in transcript order."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    read_ids: list[str] = []
    search_ids: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == "read_chunk":
                chunk_id = part.args_as_dict().get("chunk_id")
                if chunk_id:
                    read_ids.append(str(chunk_id))
            elif (
                isinstance(part, ToolReturnPart)
                and part.tool_name == "search_docs"
            ):
                search_ids.extend(_chunk_ids_from_search_output(str(part.content)))
    return read_ids, search_ids


def _dedupe(ids: list[str]) -> list[str]:
    """Order-preserving dedup."""
    seen: set[str] = set()
    ordered = []
    for chunk_id in ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _collect_sources(
    messages: Sequence[ModelMessage],
    registry: dict[str, dict[str, Any]],
) -> list[SearchResult]:
    """Deterministic source mapping: rule (a) read_chunk calls + rule (b)
    search_docs result lines, deduped, capped at MAX_SOURCES. Ids the tools
    never resolved (hallucinated by the model) have no real provenance and
    are skipped."""
    from app.query import SearchResult

    read_ids, search_ids = _source_ids_from_messages(messages)
    ordered = _dedupe(read_ids + search_ids)[:MAX_SOURCES]
    return [
        SearchResult(**registry[chunk_id])
        for chunk_id in ordered
        if chunk_id in registry
    ]


def run_agent(
    question: str,
    top_k: int = DEFAULT_AGENT_TOP_K,
    service_filter: str | None = None,
    collection_name: str | None = None,
) -> tuple[str, list[SearchResult]]:
    """One synchronous agent run: tool loop over the live stack, then the
    deterministic source mapping. Any model/infrastructure failure degrades
    to the fallback note with no sources (no retry loops)."""
    from pydantic_ai.usage import UsageLimits

    from app.vector_rag import build_retrieval_pipeline, build_vector_store

    try:
        store = build_vector_store(collection_name) if collection_name else None
        pipeline = build_retrieval_pipeline(
            top_k=top_k,
            service_filter=service_filter,
            store=store,
        )
        deps = _AgentRunDeps(pipeline=pipeline)
        agent = _build_agent(build_agent_model(), deps)
        result = agent.run_sync(
            question,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=AGENT_REQUEST_LIMIT,
                tool_calls_limit=AGENT_TOOL_CALL_LIMIT,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - model/infra failure must degrade
        logger.info("agent run failed (%s); returning fallback note.", exc)
        return AGENT_FALLBACK_NOTE, []
    return result.output or "", _collect_sources(result.all_messages(), deps.registry)
