.PHONY: help up down ps env worker sync vector-sync backend frontend test lint typecheck backup restore

help: ## List all available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':[^#]*## ' '{printf "  %-10s %s\n", $$1, $$2}'

up: ## Start infrastructure containers (Postgres, Qdrant, Neo4j)
	docker compose up -d

down: ## Stop infrastructure containers
	docker compose down

ps: ## Show infrastructure container status
	docker compose ps

env: ## Create .env from .env.example if .env is missing
	@test -f .env || cp .env.example .env

worker: ## Run the Taskiq docs-ingestion worker
	uv run python -m app.worker

sync: ## Rebuild the Neo4j knowledge graph (needs a reachable LLM)
	uv run python -m app.graph_rag sync

vector-sync: ## Rebuild the Qdrant vector index from Postgres (worker does this automatically)
	uv run python -m app.vector_rag sync

backend: ## Run the FastAPI backend with reload on port 8000
	uv run uvicorn main:app --reload --port 8000

frontend: ## Run the Streamlit chat UI on port 8501
	uv run streamlit run frontend.py --server.port 8501

test: up ## Run the test suite (starts infrastructure first)
	uv run --extra dev pytest

lint: ## Run the ruff linter
	uv run --extra dev ruff check .

typecheck: ## Run the ty type checker
	uv run --extra dev ty check

backup: ## Save the Neo4j graph to backups/
	./scripts/neo4j_backup.sh save

restore: ## Restore a Neo4j backup (usage: make restore BACKUP=backups/neo4j_<timestamp>.tar.gz)
	@test -n "$(BACKUP)" || { echo "make restore BACKUP=backups/neo4j_<timestamp>.tar.gz"; exit 1; }
	./scripts/neo4j_backup.sh restore $(BACKUP)