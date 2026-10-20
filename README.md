# Microservices Documentation RAG Assistant

A production-grade, self-contained local prototype of a microservices documentation RAG assistant built with **Python 3.14**, **FastAPI**, **PostgreSQL with `pgvector`**, **Taskiq**, **Streamlit**, and **multilingual-e5-small**.

---

## 🏗️ Architecture & Design Decisions

```mermaid
flowchart LR
    Docs["docs_mock/*.md"] --> Worker["Taskiq Worker (Sync / Hash check)"]
    Worker --> E5_Enc["sentence-transformers (E5-small, Unit Norm)"]
    E5_Enc --> PG[("PostgreSQL 18 + pgvector\n(HNSW vector_ip_ops)")]
    
    UI["Streamlit UI\n(frontend.py)"] --> API["FastAPI /chat\n(main.py)"]
    API --> RAG["RAG Service\n(rag_service.py)"]
    RAG --> PG
    API --> LLM["Local LLM (Ollama / OpenAI-compatible)"]
```

1. **Normalized Dot-Product Search (`vector_ip_ops`)**:
   `intfloat/multilingual-e5-small` embeddings are normalized to unit sphere length ($\|v\|_2 = 1.0$). For unit vectors, dot product is mathematically equivalent to cosine similarity but executes significantly faster on PostgreSQL HNSW indexes without runtime trigonometric operations.
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
- **PostgreSQL 18** with `pgvector` extension:
  ```bash
  # Using Docker
  docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_docs_db pgvector/pgvector:pg18
  ```
- *(Optional)* **Ollama** running locally with `llama3.2` or any OpenAI-compatible API (`http://localhost:11434/v1`).

---

### 2. Environment Configuration
Create a `.env` file (or edit [app/config.py](file:///Users/viktor/Projects/python_sandbox/rag_docs_service/app/config.py)):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_docs_db
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
BACKEND_API_URL=http://localhost:8000
```

---

### 3. Run Tests
Verify the entire pipeline using `pytest` via `uv`:
```bash
uv run --extra dev pytest
```

All 7 unit and smoke tests verify:
- Deterministic markdown paragraph chunking & whitespace filtering.
- Embedding generation (384 dimensions and unit norm $\|v\| = 1.0$).
- Semantic dot-product ranking between billing and auth queries.
- Context generation with microservice provenance metadata.

---

### 4. Running the Services

#### Option 1: One-Command Startup with Docker Compose (Recommended)
Start PostgreSQL 18 (`pgvector`), the Taskiq background worker, the FastAPI backend, and the Streamlit UI with a single command:
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

