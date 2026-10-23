# syntax=docker/dockerfile:1
#
# Build context is the repository root:
#   docker build -f docker/Dockerfile -t mlops-final-project:0.1.0 .
#
# Three final targets, from one file -- one for development, two for
# production:
#
#   development   the dev loop: dev dependency group (pytest, ruff,
#                 pre-commit), the project installed editable, source
#                 bind-mounted, uvicorn --reload. Never published.
#   runtime       (default) weights baked in. Starts offline, answers
#                 immediately, PRODLLM_ALLOW_DOWNLOAD=false.
#   runtime-thin  no weights. Smaller image, but the first start downloads
#                 ~470 MB from Hugging Face and /ask returns 503 until it
#                 finishes. Built with --target runtime-thin.
#
# Development and runtime are separated deliberately, and not just for size:
# a runtime image carrying pytest, ruff and a writable source tree ships test
# fixtures and a toolchain into production as attack surface. The two targets
# share this file -- and therefore the same Python version, the same locked
# dependency resolution and the same libgomp1 fix -- so the split costs no
# drift between what is tested and what is deployed.
#
# Every target runs as the unprivileged appuser (UID 10001), including
# development: a dev container that runs as root writes root-owned files into
# a bind-mounted working tree, and a dev loop that only works as root is one
# that stops working the moment it is promoted to runtime.
#
# The runtime pair is what makes the report's image-size comparison a
# measurement rather than an estimate.

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

# ----------------------------------------------------------------- development
# The development image. Not a production artifact and not a base for one:
#
#   docker build -f docker/Dockerfile --target development -t mlops-final-project:0.1.0-dev .
#   docker compose --profile dev up dev            # with src/ and tests/ mounted
#   docker compose --profile dev run --rm dev pytest
#
# It keeps uv (so `uv sync` and `uv run` work inside the container) where the
# runtime images deliberately do not: the runtime stages copy a finished
# /app/.venv out of the builder and ship no package manager at all.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS development

# libgomp1 for the same reason runtime-base installs it -- faiss-cpu's wheel
# links libgomp.so.1 and the slim images do not ship it. git additionally,
# because pre-commit refuses to run outside a repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# UV_CACHE_DIR and HF_HOME are redirected under /app, which appuser owns:
# their defaults sit in root's home, so `uv run` and the first model download
# would fail with a permission error the moment the container stopped running
# as root.
#
# No PRODLLM_* here, deliberately -- unlike runtime-base, which bakes them.
# tests/test_config.py asserts the shipped defaults (artifacts_dir ==
# Path("artifacts"), log_level == "INFO"), and pydantic-settings reads the
# environment, so a baked PRODLLM_ARTIFACTS_DIR fails two tests inside the very
# image built to run them. Nothing is lost: artifacts_dir is relative by
# design and WORKDIR is /app, so it already resolves to /app/artifacts.
# Dev-time values belong in docker-compose.yml, where they are visible and
# overridable per run.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CACHE_DIR=/app/.cache/uv \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# The user is created *before* the virtualenv exists, and everything below
# runs as that user. Creating it last and fixing ownership with a
# `chown -R /app` afterwards works, but it rewrites all ~9000 files of the
# ~1 GB dependency tree into a second copy of itself in a new layer -- the
# image then carries the venv twice. Here uv writes files that are correctly
# owned to begin with, so no layer has to be rewritten.
RUN useradd --no-create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/.cache/uv /app/.cache/huggingface \
    && chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# Same two-step dependency layering as the builder, and no --no-default-groups:
# the dev group (pytest, pytest-cov, ruff, pre-commit) is exactly what this
# image exists for.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
# The cache is a build mount, not a layer. Without it uv's downloaded wheels
# are baked in next to the venv they were copied into -- the dev image measured
# 3.45 GB with the cache inside and 2.07 GB without, for an identical venv,
# because UV_LINK_MODE=copy means every wheel is stored twice. uid/gid are
# required: the mount is created by the builder, and appuser must be able to
# write it.
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    uv sync --locked --no-install-project

# Editable, unlike the runtime images: the package resolves through a .pth file
# pointing at /app/src, which compose bind-mounts from the host -- so an edit
# on the host is what uvicorn --reload picks up, with no rebuild per edit.
COPY --chown=appuser:appuser README.md ./
COPY --chown=appuser:appuser src ./src
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    uv sync --locked

# Copied so the image is usable on its own (`docker run ... pytest`); compose
# mounts the host copies over them for the edit-test loop. openapi.json comes
# along because tests/test_openapi_export.py asserts the committed spec still
# matches the app -- without it the suite fails in the container only.
COPY --chown=appuser:appuser tests ./tests
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser openapi.json ./

EXPOSE 8000

# No HEALTHCHECK: a dev container is routinely started with no index and no
# weights on purpose, and `unhealthy` on a container that is behaving exactly
# as intended trains you to ignore the column.
#
# --reload-dir is restricted to the source tree, or the reloader also watches
# /app/artifacts and restarts the server mid-download when weights arrive.
CMD ["uvicorn", "prodllm.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/app/src"]

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
