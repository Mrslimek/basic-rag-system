# Microservices Documentation RAG Assistant

A production-grade, self-contained local prototype of a microservices documentation RAG assistant built with **Python 3.14**, **FastAPI**, **LlamaIndex**, **PostgreSQL**, **Qdrant**, **Neo4j**, **Taskiq**, **Streamlit**, and **multilingual-e5-small**.

---

## 🏗️ Architecture & Design Decisions

```mermaid
flowchart LR
    Docs["docs_mock/*.md"] --> Worker["Taskiq Worker (Sync / Hash check)"]
    Worker --> PG[("PostgreSQL\n(documents + graph JSONB)")]
    Worker --> QD[("Qdrant\n(doc_chunks vectors)")]
    Worker --> NEO[("Neo4j\n(service graph)")]

    UI["Streamlit UI\n(frontend.py)"] --> API["FastAPI /chat\n(main.py)"]
    API --> RAG["LlamaIndex pipeline\n(app/llama_rag.py)"]
    RAG --> QD
    RAG --> LLM["Local LLM (Ollama / OpenAI-compatible)"]
```

1. **Single LlamaIndex retrieval pipeline**: `/chat` runs one path — a LlamaIndex retriever over the Qdrant `doc_chunks` collection with LLM synthesis (`app/llama_rag.aquery`). Postgres remains the source of truth for documents and provenance.
2. **Asymmetric E5 Retrieval**:
   - Stored document chunks are prefixed with `"passage: "`.
   - Incoming user questions are prefixed with `"query: "`.
   This aligns vectors in the contrastive embedding subspace learned by the E5 model family.
3. **Deterministic Paragraph Chunking**:
   Technical markdown specifications contain structured endpoint descriptions and JSON/YAML code examples. Splitting by paragraphs (`\n\n+`) prevents cutting schema bodies or tables midway.
4. **Taskiq Background Worker**:
   Scans `docs_mock/*.md` periodically, computes SHA-256 hashes to detect updates, and atomically re-indexes modified microservice documentation.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.14** (configured via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** (for PostgreSQL, Qdrant, and Neo4j via docker-compose)
- *(Optional)* **Ollama** running locally with `llama3.2` or any OpenAI-compatible API (`http://localhost:11434/v1`).

---

### 2. Environment Configuration
Create a `.env` file (or edit [app/config.py](file:///Users/viktor/Projects/python_sandbox/rag_docs_service/app/config.py)):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_docs_db
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=doc_chunks
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
BACKEND_API_URL=http://localhost:8000
```

---

### 3. Run Tests
Verify the pipeline using `pytest` via `uv`:
```bash
uv run --extra dev pytest
```

---

### 4. Running the Services

#### Option 1: One-Command Startup with Docker Compose (Recommended)
Start PostgreSQL, Qdrant, Neo4j, the Taskiq background worker, the FastAPI backend, and the Streamlit UI with a single command:
```bash
docker compose up --build
```
- **Streamlit Chat UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- *Note: The `api` container automatically connects to a local Ollama instance on your host machine via `http://host.docker.internal:11434/v1`.*

---

#### Option 2: Running Locally via `uv`

1. **Start Taskiq Worker (Syncs & Embeds Markdown Docs)**:
   ```bash
   uv run python -m app.worker
   ```
2. **Start FastAPI Backend**:
   ```bash
   uv run uvicorn main:app --reload --port 8000
   ```
3. **Start Streamlit Chat UI**:
   ```bash
   uv run streamlit run frontend.py --server.port 8501
   ```
