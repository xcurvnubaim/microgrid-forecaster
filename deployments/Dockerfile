# Multi-stage build using uv for fast, reproducible installs.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# Install deps first (cached layer) from lockfile.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the project itself.
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Serve: scheduler + Redis publisher + HTTP API.
CMD ["forecaster", "serve", "--config", "configs/forecaster.yaml"]
