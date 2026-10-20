FROM python:3.14-slim

# Copy uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and copy mode for Docker mounts
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# 1. Install dependencies first (cached layer unless pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. Pre-download the multilingual-e5-small embedding model during image build
# so worker and api containers start instantaneously without re-downloading at runtime
RUN uv run --no-project python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# 3. Copy application source code and install project
COPY . .
RUN uv sync --frozen --no-dev

# Add virtualenv binaries to PATH
ENV PATH="/app/.venv/bin:$PATH"
