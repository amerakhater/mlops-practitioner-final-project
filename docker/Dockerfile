# Build context is the repository root:
#   docker build -f docker/Dockerfile -t mlops-final-project .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies and the project itself are installed from the lock file,
# so the image matches what `uv sync` produces locally.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "prodllm.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
