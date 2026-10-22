# syntax=docker/dockerfile:1
#
# Build context is the repository root:
#   docker build -f docker/Dockerfile -t mlops-final-project:0.1.0 .
#
# Two final targets, from one file:
#
#   runtime       (default) weights baked in. Starts offline, answers
#                 immediately, PRODLLM_ALLOW_DOWNLOAD=false.
#   runtime-thin  no weights. Smaller image, but the first start downloads
#                 ~470 MB from Hugging Face and /ask returns 503 until it
#                 finishes. Built with --target runtime-thin.
#
# The pair is what makes the report's image-size comparison a measurement
# rather than an estimate.

# ---------------------------------------------------------------- dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Third-party dependencies resolve in their own layer, before any source is
# copied, so editing a route rebuilds one cheap layer instead of
# re-downloading torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project

# --no-editable installs prodllm as a real package, so /app/src is a build-time
# input only and never reaches the runtime image.
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

# -------------------------------------------------------------------- weights
# Only built when the final target needs it, so `--target runtime-thin` never
# pays for the download.
FROM builder AS weights

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/tmp/hf

COPY scripts/fetch_weights.py ./scripts/
RUN python scripts/fetch_weights.py --artifacts-dir /weights-artifacts

# ---------------------------------------------------------------- runtime base
FROM python:3.12-slim-bookworm AS runtime-base

# faiss-cpu's wheel dynamically links libgomp.so.1, which this image does not
# ship. Without it the container starts, logs "index unavailable" and serves
# 503 forever, with the real ImportError buried in the startup log -- a failure
# that looks exactly like a missing artifact.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PRODLLM_ARTIFACTS_DIR=/app/artifacts \
    PRODLLM_LOG_LEVEL=INFO \
    PRODLLM_TORCH_NUM_THREADS=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# The vector store ships inside the image: a flat FAISS index over ~1100
# article chunks is ~6 MB, needs no server and loads in milliseconds, so there
# is nothing to gain from a database container. Built by
# scripts/build_index.py and gitignored, so it must exist locally before the
# build -- see the README.
COPY artifacts/index /app/artifacts/index

RUN useradd --no-create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/artifacts
USER appuser

EXPOSE 8000

# 200 is not enough: /health answers 200 while degraded on purpose, so that a
# container which cannot load its model is diagnosable instead of
# crash-looping. Readiness is documents_indexed > 0 with the model loaded.
# No curl in a slim image, and urllib is stdlib, so no dependency is added.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); sys.exit(0 if d['model_loaded'] and d['documents_indexed'] > 0 else 1)"]

CMD ["uvicorn", "prodllm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------- thin (no model)
FROM runtime-base AS runtime-thin

# The Hugging Face cache must be writable by UID 10001 and should be a mounted
# volume, or every container restart re-downloads the model.
ENV PRODLLM_ALLOW_DOWNLOAD=true \
    HF_HOME=/app/.cache/huggingface
USER root
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache
USER appuser

# A cold start includes the download, so the healthcheck must not mark the
# container unhealthy while it is still fetching weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); sys.exit(0 if d['model_loaded'] and d['documents_indexed'] > 0 else 1)"]

LABEL org.opencontainers.image.title="Arabic legal document Q&A (thin)" \
      org.opencontainers.image.description="Weights are downloaded on first start; mount a volume at /app/.cache/huggingface." \
      org.opencontainers.image.source="https://github.com/amerakhater/mlops-final-project"

# ----------------------------------------------------------- default (weights)
# Last stage = the default target, so a plain `docker build` produces the
# self-contained image.
FROM runtime-base AS runtime

COPY --from=weights /weights-artifacts/model /app/artifacts/model
COPY --from=weights /weights-artifacts/metadata.json /app/artifacts/metadata.json

# Nothing is fetched at runtime. If the weights are somehow absent, startup
# fails loudly rather than silently reaching out to Hugging Face, which would
# make the image's provenance unverifiable.
ENV PRODLLM_ALLOW_DOWNLOAD=false

LABEL org.opencontainers.image.title="Arabic legal document Q&A" \
      org.opencontainers.image.description="Retrieval-augmented answers over the Egyptian Civil Code, with article citations. Self-contained: weights and FAISS index are baked in." \
      org.opencontainers.image.source="https://github.com/amerakhater/mlops-final-project"
