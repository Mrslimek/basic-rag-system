import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db_session, init_db
from app.llm import get_chat_model
from app.rag_service import rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info("Verifying database schema on FastAPI startup...")
    try:
        await init_db()
        logger.info("Database schemas and pgvector initialized successfully.")
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL on startup ({e}). "
            "Ensure PostgreSQL with pgvector is running before invoking RAG queries."
        )
    yield


app = FastAPI(
    title="Microservices Documentation RAG Assistant",
    version="0.1.0",
    description="Vector search and RAG assistant over microservice markdown specifications using pgvector and e5-small.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query about microservices")
    service_filter: str | None = Field(
        None, description="Optional microservice name to narrow vector search (e.g., 'billing')"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context passages to retrieve")


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
    Sends retrieved context and prompt to local Ollama or OpenAI-compatible endpoint.
    Falls back gracefully to a structured synthesis if the local LLM is offline.
    """
    system_instruction = (
        "You are an expert microservices architecture assistant. "
        "Answer the user's question using ONLY the provided documentation context below. "
        "Always cite the microservice name and endpoint paths accurately. "
        "If the answer cannot be found in the context, explicitly state that."
    )

    full_user_message = f"Documentation Context:\n{context}\n\nQuestion:\n{prompt}"

    llm = get_chat_model(temperature=0.2)
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_instruction),
                HumanMessage(content=full_user_message),
            ]
        )
        return response.text
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
    Primary RAG endpoint:
    1. Embeds user query with 'query: ' prefix and unit normalization.
    2. Performs inner-product similarity search against pgvector.
    3. Formats provenance context with microservice metadata.
    4. Calls LLM (or fallback) and returns structured response.
    """
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
