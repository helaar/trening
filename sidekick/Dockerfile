# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

# libgomp1 is required at runtime by onnxruntime (pulled in via crewai/chromadb).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# --- dependency layers, cached independently of application source ---

FROM base AS deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

FROM base AS deps-dev
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# --- dev target: used by docker-compose.override.yml with src/ bind-mounted ---

FROM deps-dev AS dev
COPY src ./src
COPY scripts ./scripts
COPY logging.yaml ./logging.yaml
EXPOSE 5175
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5175", "--reload", "--reload-dir", "/app/src"]

# --- runtime target: production image, no dev dependencies, non-root ---

FROM deps AS runtime
COPY src ./src
COPY scripts ./scripts
COPY logging.yaml ./logging.yaml
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 5175
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5175/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5175"]
