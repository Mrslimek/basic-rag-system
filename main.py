import logging
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import Depends, FastAPI, HTTPException
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db_session, init_db
from app.llm import SYSTEM_INSTRUCTION, get_chat_model
from app.rag_service import rag_service
from app.vectorstore import get_vectorstore
from app import llama_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info("Verifying database schema and Qdrant collection on FastAPI startup...")
    try:
        await init_db()
        logger.info("Database schemas initialized successfully.")
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL on startup ({e}). "
            "Ensure PostgreSQL is running before invoking RAG queries."
        )
    try:
        store = get_vectorstore()
        await store.ensure_collection()
        logger.info("Qdrant collection is ready.")
    except Exception as e:
        logger.warning(
            f"Could not reach Qdrant on startup ({e}). "
            "Run `docker compose up -d qdrant` before invoking RAG queries."
        )
    yield


app = FastAPI(
    title="Microservices Documentation RAG Assistant",
    version="0.1.0",
    description="Vector search and RAG assistant over microservice markdown specifications using Qdrant (vectors), Neo4j (graph) and e5-small embeddings.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query about microservices")
    service_filter: str | None = Field(
        None, description="Optional microservice name to narrow vector search (e.g., 'billing')"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context passages to retrieve")
    mode: Literal["handwritten", "llamaindex"] = Field(
        default="llamaindex",
        description="Retrieval pipeline: hand-written Qdrant+SQL path, or the LlamaIndex framework path",
    )


class SourceReference(BaseModel):
    service_name: str
    doc_file_path: str
    chunk_index: int
    similarity_score: float
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[SourceReference]


async def query_llm(prompt: str, context: str) -> str:
    """
    Sends retrieved context and prompt to the local OpenAI-compatible endpoint
    through the shared LlamaIndex client. Falls back gracefully to a structured
    synthesis if the local LLM is offline.
    """
    full_user_message = f"Documentation Context:\n{context}\n\nQuestion:\n{prompt}"

    llm = get_chat_model(temperature=0.2)
    try:
        response = await llm.achat(
            [
                ChatMessage(role="system", content=SYSTEM_INSTRUCTION),
                ChatMessage(role="user", content=full_user_message),
            ]
        )
        # response.message is a ChatMessage; take its .content.
        # str(ChatMessage) would prepend "assistant: " — a role label, not text.
        return response.message.content
    except Exception as exc:
        logger.info(f"Local LLM not reachable at {settings.LLM_API_BASE} ({exc}). Using direct context synthesis.")

    # Graceful fallback: return the retrieved contextual knowledge directly
    return (
        "*(Note: Local LLM was unreachable; displaying retrieved documentation matches directly)*\n\n"
        f"{context}"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.EMBEDDING_MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Primary RAG endpoint. Two interchangeable pipelines behind `mode`:
      handwritten — our own Qdrant ANN + Postgres hydration (baseline, A/B reference)
      llamaindex  — the framework path: IngestionPipeline-built collection,
                    RetrieverQueryEngine synthesis, same embeddings & LLM.
    Both return the same reply/sources shape for direct comparison.
    """
    if request.mode == "llamaindex":
        try:
            llm_reply, results = await llama_rag.aquery(
                question=request.message,
                top_k=request.top_k,
                service_filter=request.service_filter,
            )
        except Exception as e:
            logger.error(f"Error during LlamaIndex query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LlamaIndex retrieval error: {str(e)}")
    else:
        try:
            results = await rag_service.retrieve_chunks(
                session=session,
                query=request.message,
                top_k=request.top_k,
                service_filter=request.service_filter,
            )
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database retrieval error: {str(e)}")

        context_str = rag_service.build_context(results)
        llm_reply = await query_llm(prompt=request.message, context=context_str)

    sources = [
        SourceReference(
            service_name=r.service_name,
            doc_file_path=r.doc_file_path,
            chunk_index=r.chunk_index,
            similarity_score=round(r.similarity_score, 4),
            snippet=r.content[:200] + ("..." if len(r.content) > 200 else ""),
        )
        for r in results
    ]

    return ChatResponse(reply=llm_reply, sources=sources)
