import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.database import init_db
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
        # ensure_collection is blocking sync I/O; keep it off the event loop.
        await asyncio.to_thread(llama_rag.ensure_collection, settings.QDRANT_COLLECTION)
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
    retrieve_mode: Literal["vector", "graph"] = Field(
        default="vector",
        description="RAG branch: 'vector' = Qdrant semantic search, 'graph' = LlamaIndex PropertyGraphIndex over Neo4j",
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


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.EMBEDDING_MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """RAG endpoint: LlamaIndex retrieval and LLM synthesis over the Qdrant collection."""
    try:
        llm_reply, results = await llama_rag.aquery(
            question=request.message,
            top_k=request.top_k,
            service_filter=request.service_filter,
            retrieve_mode=request.retrieve_mode,
        )
    except Exception as e:
        logger.error(f"Error during LlamaIndex query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LlamaIndex retrieval error: {str(e)}")

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
