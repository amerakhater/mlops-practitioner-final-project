# mlops-final-project

A FastAPI service skeleton with structured JSON logging and per-request
correlation IDs. This document covers how to run it locally and how to deploy
it as a container.

> **Current state:** the application exposes no business routes yet. Only
> FastAPI's built-in endpoints (`/docs`, `/redoc`, `/openapi.json`) respond;
> every other path returns `404`. The request logging and correlation-ID
> middleware are fully wired and work on every request, including the 404s.

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.12 or newer — manages Python and dependencies
- Python 3.12 (uv installs it automatically from `.python-version`)
- Docker, only for the container deployment

You do not need a pre-existing Python installation or a manually created
virtualenv; `uv` handles both.

## Install

```bash
cd mlops-final-project
uv sync
```

`uv sync` creates `.venv/` and installs the exact versions pinned in
`uv.lock`. Re-run it after pulling changes that touch `pyproject.toml` or
`uv.lock`.

## Run locally

```bash
uv run uvicorn prodllm.api.main:app --reload --port 8000
```

- `--reload` restarts the server on source changes. Use it for development only.
- The ASGI application object is `app` in
  [src/prodllm/api/main.py](src/prodllm/api/main.py).

The service is then on <http://127.0.0.1:8000>, with interactive API docs at
<http://127.0.0.1:8000/docs>.

### Verify it is working

```bash
curl -i http://127.0.0.1:8000/openapi.json
```

Two things confirm the middleware is active:

1. The response carries an `x-request-id` header with a fresh UUID per request.
2. The server prints one JSON log line per request, for example:

```json
{"timestamp": "2026-10-21T14:09:07.605150+00:00", "level": "INFO", "logger": "prodllm.api", "message": "Request completed", "correlation_id": "c187015d-27dd-4b12-96cb-623ace68cbe3", "method": "GET", "path": "/openapi.json", "status_code": 200, "latency_ms": 1.16}
```

The `correlation_id` in the log matches the `x-request-id` header, so a client
can quote the header value and you can find the exact request in the logs.

## Run as a container

The image is built from [docker/Dockerfile](docker/Dockerfile). **Build from
the repository root**, not from inside `docker/` — the build context must
contain `src/`, `pyproject.toml`, `uv.lock` and `README.md`:

```bash
docker build -f docker/Dockerfile -t mlops-final-project:latest .
docker run --rm -p 8000:8000 mlops-final-project:latest
```

The image installs dependencies from `uv.lock` with `uv sync --locked --no-dev`,
so a build fails loudly if the lock file is out of date rather than silently
resolving different versions. It runs as the non-root user `appuser` (UID
10001) and listens on port 8000.

To pass configuration in, use `--env-file`:

```bash
docker run --rm -p 8000:8000 --env-file .env mlops-final-project:latest
```

`.env` is excluded from the image by `.dockerignore`, so secrets are injected
at run time and never baked into a layer.

## Deploy

The container is a standard stateless HTTP service: it listens on `$PORT`-style
port 8000, holds no local state, and logs JSON to stdout. That makes it
deployable as-is to any container host (Cloud Run, ECS, Azure Container Apps,
Kubernetes, a plain Docker host).

Points that matter when promoting it beyond a laptop:

- **Drop `--reload`.** The image's default command already omits it.
- **Workers.** The default command runs a single uvicorn worker. Scale by
  running more container replicas rather than more in-process workers — that
  keeps per-instance memory predictable and lets the orchestrator handle
  restarts.
- **Logs.** JSON goes to stdout, which is what container log collectors expect.
  Do not add file logging; let the platform capture the stream.
- **Health checks.** There is no health endpoint yet. Until one exists, point
  liveness/readiness probes at `/openapi.json`, which returns `200`. Replacing
  that with a real `/health` route is the first thing worth adding.
- **Port.** Override with `--port` in the command if your platform requires a
  specific one.

Example, pinning a version tag and overriding the port:

```bash
docker build -f docker/Dockerfile -t mlops-final-project:0.1.0 .
docker run --rm -p 8080:8080 mlops-final-project:0.1.0 \
  uvicorn prodllm.api.main:app --host 0.0.0.0 --port 8080
```

## Configuration

`.env` and `.env.example` exist but are currently empty, as is
[src/prodllm/config.py](src/prodllm/config.py). The
application reads no environment variables yet — the only runtime knobs today
are the uvicorn flags above. As settings are added, document them in
`.env.example` and keep real values out of `.env` in version control.

## Project layout

```
mlops-final-project/
├── docker/Dockerfile                        container image definition
├── src/prodllm/
│   ├── api/main.py                          FastAPI app object and startup wiring
│   ├── api/middleware.py                    correlation ID + request logging
│   ├── logging_conf.py                      JSON log formatter and filter
│   └── config.py                            settings (empty placeholder)
├── tests/                                   empty placeholder
├── pyproject.toml                           project metadata and dependencies
└── uv.lock                                  pinned dependency versions
```

## Tests

`tests/` is an empty placeholder — there is no test suite and no test runner
configured in `pyproject.toml` yet. Once tests are added, run them with
`uv run pytest`.
