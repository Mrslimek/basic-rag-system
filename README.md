# Microservices Documentation RAG Assistant

A production-grade, self-contained local prototype of a microservices documentation RAG assistant built with **Python 3.14**, **FastAPI**, **LlamaIndex**, **PostgreSQL**, **Qdrant**, **Neo4j**, **Taskiq**, **Streamlit**, and **multilingual-e5-small**.

---

## 🏗️ Architecture & Design Decisions

```mermaid
flowchart LR
    Docs["docs_mock/*.md"] -->|"every 60s\nSHA-256 hash check"| Worker["Taskiq Worker\n(app/worker.py)"]
    Worker --> PG[("PostgreSQL\nchunk text = source of truth")]
    Worker -->|"IngestionPipeline"| QD[("Qdrant\ncollection: doc_chunks")]

    PG -->|"app.graph_rag sync\nLLM relation extraction"| NEO[("Neo4j\nknowledge graph")]

    UI["Streamlit UI\n(frontend.py)"] --> API["FastAPI /chat\n(main.py)"]
    API --> RAG["app/llama_rag.py\none LlamaIndex flow"]
    RAG -->|"retrieve_mode = vector"| QD
    RAG -->|"retrieve_mode = graph"| NEO
    RAG -->|"retrieve_mode = hybrid\n(vector-first)"| QD
    RAG -->|"retrieve_mode = hybrid\n(graph facts)"| NEO
    RAG --> LLM["LLM synthesis\n(Ollama / OpenAI-compatible)"]
```

1. **One LlamaIndex flow, three retrieval modes**: `/chat` takes `retrieve_mode: "vector" | "graph" | "hybrid"`. Vector mode queries the Qdrant `doc_chunks` collection; graph mode queries a LlamaIndex `PropertyGraphIndex` over Neo4j. All three branches share the same response synthesizer and LLM client, so switching retrieval strategies never changes how answers are composed. Hybrid mode retrieves the same Qdrant chunks as vector mode and appends a `Graph facts` block — the mentioned entities and their `X -> REL -> Y` semantic triplets from Neo4j — to each chunk's context text. The mode selection lives entirely in `app/llama_rag.py`; nothing else knows which branch ran.
2. **Postgres is the source of truth for chunk text**: the worker persists every document and its chunks in PostgreSQL first. Qdrant and Neo4j are derived stores that can always be rebuilt from Postgres.
3. **Asymmetric E5 retrieval**:
   - Stored document chunks are prefixed with `"passage: "`.
   - Incoming user questions are prefixed with `"query: "`.
   This aligns vectors in the contrastive embedding subspace learned by the E5 model family. Prefixing is applied once in `app/embeddings.py`, so every indexing and query path produces interchangeable vectors.
4. **Deterministic paragraph chunking**: technical markdown specifications contain structured endpoint descriptions and JSON/YAML code examples. Splitting by paragraphs (`\n\n+`) prevents cutting schema bodies or tables midway.
5. **Taskiq background worker**: a scheduler triggers every 60 seconds, scans `docs_mock/*.md`, computes SHA-256 hashes to detect changes, replaces changed documents' chunks in Postgres atomically, and re-embeds them into Qdrant through the LlamaIndex `IngestionPipeline`.
6. **The knowledge graph is built separately from the vector index**: a full rebuild is triggered on demand with `uv run python -m app.graph_rag sync`. It reads chunks from Postgres and extracts entities and relationships with an LLM inside the index via LlamaIndex's `SchemaLLMPathExtractor`, writing the result to Neo4j. Retrieval must run against an APOC-enabled Neo4j; `docker-compose.yml` provisions it with `NEO4J_PLUGINS: '["apoc"]'`.

---

## 🔄 Data Flows

**Ingestion — automatic, every 60 seconds (worker):**

```
docs_mock/*.md → hash check → paragraph chunks → PostgreSQL → IngestionPipeline → Qdrant (doc_chunks)
```

**Graph rebuild — manual, on demand:**

```
uv run python -m app.graph_rag sync → read chunks from PostgreSQL → LLM extraction (SchemaLLMPathExtractor) → Neo4j (full rebuild)
```

**Query — per request:**

```
Streamlit sidebar (vector | graph | hybrid) → POST /chat { retrieve_mode } → LlamaIndex retriever (Qdrant, Neo4j, or both) → shared LLM synthesis → answer + sources
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.14** (configured via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** (for PostgreSQL, Qdrant, and Neo4j via docker-compose)
- *(Optional)* **llama.cpp** running locally serving `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` at `http://127.0.0.1:8080/v1` (API key `ollama`), or any OpenAI-compatible API. Graph sync and answer synthesis both call the LLM, so one must be reachable for those steps.

---

### 2. Environment Configuration
Create a `.env` file (or edit [app/config.py](file:///Users/viktor/Projects/python_sandbox/rag_docs_service/app/config.py)):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_docs_db
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=doc_chunks
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag123
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
LLM_API_BASE=http://127.0.0.1:8080/v1
LLM_API_KEY=ollama
LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
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
Start PostgreSQL, Qdrant (v1.20), the APOC-enabled Neo4j 5.26, the Taskiq background worker, the FastAPI backend, and the Streamlit UI with a single command:
```bash
docker compose up --build
```
- **Streamlit Chat UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- *Note: The `api` container automatically connects to a local Ollama instance on your host machine via `http://host.docker.internal:11434/v1`.*

#### Option 2: Running Locally via `uv`

1. **Start the infrastructure** (Postgres, Qdrant, Neo4j):
   ```bash
   docker compose up -d
   ```
2. **Start the Taskiq Worker** (syncs and embeds markdown docs every 60 seconds):
   ```bash
   uv run python -m app.worker
   ```
3. **Start the FastAPI backend**:
   ```bash
   uv run uvicorn main:app --reload --port 8000
   ```
4. **Start the Streamlit chat UI**:
   ```bash
   uv run streamlit run frontend.py --server.port 8501
   ```

---

### 5. Building the Knowledge Graph
Vector search works as soon as the worker has indexed the docs. The graph branch needs one explicit build step, which fully rebuilds the Neo4j graph from the chunks stored in Postgres using LLM-based extraction:
```bash
uv run python -m app.graph_rag sync
```
Re-run it whenever documentation changes materially; it wipes and rebuilds the graph in place.
