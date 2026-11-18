# Arabic legal document Q&A — Egyptian Civil Code

Ask a question in Arabic or English about the Egyptian Civil Code and get an
answer **with the article citations it was drawn from**. Retrieval-augmented:
the Code is parsed into one record per article, every article is embedded, and
the answer quotes only what was retrieved.

Citations are the point. An answer a lawyer cannot check against an article
number is worthless, so every response carries the article number, the printed
citation, the source page and the retrieval score.

## Quickstart — three commands

```bash
docker pull amerakhater/mlops-final-project:0.1.0
docker run --rm -p 8000:8000 amerakhater/mlops-final-project:0.1.0
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"ما هي مدة التقادم في القانون المدني؟"}'
```

The second command stays in the foreground, so run the third in a second
terminal. Nothing else is needed: the model weights and the vector index are
both inside the image, so there is no download, no API key and no database.
Startup is about 20 seconds, almost all of it loading the encoder.

On Windows, `curl` in PowerShell is an alias for `Invoke-WebRequest` and takes
different flags. Use `curl.exe` explicitly, or:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
  -ContentType 'application/json' `
  -Body ([Text.Encoding]::UTF8.GetBytes('{"question":"ما هي مدة التقادم في القانون المدني؟"}'))
```

## What it answers over

| | |
|---|---|
| Source | Egyptian Civil Code (القانون المدني المصري), Law 131 of 1948, bilingual Arabic/English PDF |
| Articles | 1149, of which 56 are flagged repealed and 1091 are bilingual |
| Indexed chunks | 1092, one per article (long articles split on sub-clauses) |
| Vectors | 2183 — one per language per chunk, so Arabic and English questions both hit |
| Index | FAISS `IndexFlatIP` over unit-norm vectors, ~6 MB, no server |
| Encoder | `intfloat/multilingual-e5-small`, 384 dims, mean pooling + L2 norm |

Repealed articles are **flagged, not deleted**: a question about a repealed
provision should be answerable with the fact that it was repealed, which is
impossible if the record is gone.

## Two serving surfaces

There are two ways to serve this system, and both drive the same
`prodllm.rag.RagEngine` — retrieval, citation and refusal behaviour is defined
once.

| | FastAPI (`:8000`) | BentoML (`:3000`) |
|---|---|---|
| Entry point | `prodllm.api.main:app` | `prodllm.serving.service:LegalRagService` |
| Contract | committed [openapi.json](openapi.json), gated in CI | `bentofile.yaml` |
| Streaming | no | **yes** — `/ask_stream` |
| Batch re-indexing | no | **yes** — `/reindex`, no restart |
| Embedding endpoints | `/predict`, `/predict/batch` | no |
| Canary identity | no | **yes** — `release_channel` |

Use FastAPI for the embedding primitives and the published contract; use the
Bento for production question answering. The split is explained in
[reports/module-3.md](reports/module-3.md) — in short, the behaviour lives in
`prodllm/serving/engine.py`, which imports no `bentoml` and is therefore
testable with no server and no port, while `service.py` is request parsing and
status codes only.

## FastAPI endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | A question in, an answer plus article citations out |
| `GET` | `/health` | Liveness and readiness (`documents_indexed`) |
| `GET` | `/metadata` | What is actually loaded: model id, dim, format, SHA-256 |
| `POST` | `/predict` | Embed one text — the retrieval primitive |
| `POST` | `/predict/batch` | Embed up to 64 texts in one forward pass |

Interactive docs at <http://localhost:8000/docs>. The same contract is
committed as [openapi.json](openapi.json) — 3.1.0, generated from the
application, not written by hand:

```bash
uv run python scripts/export_openapi.py            # write / refresh it
uv run python scripts/export_openapi.py --check    # fail if it is stale
```

A committed spec is what makes a contract change visible in a diff, and lets a
client be generated or the API imported into a collection without a running
container. The export also runs as a pre-commit hook whenever something under
`src/prodllm/api/` changes — it rewrites the file and the hook fails, the same
way `ruff --fix` does — so the spec cannot drift from the routes it describes.

### `POST /ask`

```json
{"question": "ما هي مدة التقادم في القانون المدني؟"}
```

Real response, abridged — article text truncated, two of five sources dropped:

```json
{
  "answer": "Egyptian Civil Code, Article 388: A debtor cannot renounce the benefits of prescription before he has acquired the right to invoke it ...\n\nEgyptian Civil Code, Article 386: Prescription extinguishes the obligation, but leaves a natural obligation upon the debtor ...",
  "sources": [
    {
      "article_number": 388,
      "citation": "Egyptian Civil Code, Article 388",
      "source_page": 53,
      "score": 0.8476,
      "text_ar": "لا يجوز الترول عن التقادم قبل ثبوت الحق فيه ...",
      "text_en": "A debtor cannot renounce the benefits of prescription ..."
    },
    {
      "article_number": 974,
      "citation": "Egyptian Civil Code, Article 974",
      "source_page": 140,
      "score": 0.8311,
      "text_ar": "أيا كانت مدة التقادم المكسب فإنه يقف متى وجد سبب الوقف.",
      "text_en": "Acquisitive prescription, whatever its period, is suspended if any cause exists for such suspension."
    }
  ],
  "grounded": true,
  "latency_ms": 45.3
}
```

`grounded` is false when nothing was retrieved; the answer then says so and
`sources` is empty, rather than answering from the least-irrelevant article
available.

An empty or whitespace-only question is a `422`, not an empty answer.

**Answers are extractive by default.** With no LLM configured the service
quotes the retrieved articles verbatim, which cannot hallucinate. Set
`PRODLLM_LLM_BASE_URL` and `PRODLLM_LLM_MODEL` to route through an
OpenAI-compatible chat endpoint instead; the prompt instructs the model to
answer only from the supplied articles.

### `GET /health`

```json
{"status": "healthy", "documents_indexed": 1092, "version": "0.1.0",
 "model_loaded": true, "uptime_seconds": 19.45}
```

It answers `200` **even when degraded**, with `status: "degraded"` and
`documents_indexed: 0`. That is deliberate: a container that reports why it
cannot serve is far easier to diagnose than one that crash-loops before it can
log anything. Readiness is therefore `documents_indexed > 0`, which is what the
image's `HEALTHCHECK` asserts — not a bare `200`.

### `GET /metadata`

```json
{"service_version": "0.1.0", "source_model_id": "intfloat/multilingual-e5-small",
 "embedding_dim": 384, "max_seq_length": 512, "serialization_format": "safetensors",
 "artifact_sha256": "a1b2c3...", "pooling": "mean", "normalized": true,
 "supported_prefixes": ["query: ", "passage: "],
 "max_batch_size": 64, "max_text_chars": 8000}
```

It reports the artifact **actually loaded**, not what the config asked for.
`artifact_sha256` is empty when the weights came from the Hugging Face cache
rather than from a build that wrote `artifacts/metadata.json`.

### `POST /predict`

```json
{"text": "ما هي مدة التقادم في القانون المدني؟", "prefix": "query: "}
```

`prefix` is `"query: "` or `"passage: "` and nothing else — e5 is trained with
instruction prefixes, and omitting them degrades retrieval with no error. An
unsupported prefix is a `422` at the edge rather than a quietly worse vector.
Use `"passage: "` only when embedding documents to index.

## Serving with BentoML

```bash
uv sync --group serving
uv run --group serving bentoml serve prodllm.serving.service:LegalRagService --port 3000
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Asynchronous. Answer, citations, latency, release channel |
| `POST` | `/ask_stream` | The same answer, streamed as it is produced |
| `POST` | `/reindex` | Add documents to the live index, no restart |
| `POST` | `/status` | Documents indexed, and which channel answered |
| `GET` | `/readyz` | BentoML's own readiness probe |

`/status` rather than `/healthz` on purpose: BentoML already serves `/healthz`,
`/livez` and `/readyz`, and taking one of those names leaves a path where `GET`
reaches BentoML's probe and `POST` reaches application code.

```bash
curl -s localhost:3000/ask -H 'content-type: application/json' \
  -d '{"question":"What is the prescription period for obligations?"}'
```

```json
{
  "answer": "Egyptian Civil Code, Article 374: The term of prescription for obligations is fifteen years …",
  "sources": [{"article_number": 374, "citation": "Egyptian Civil Code, Article 374", "score": 0.9049, "…": "…"}],
  "grounded": true,
  "latency_ms": 48.6,
  "release_channel": "stable",
  "service_version": "0.1.0"
}
```

### Streaming tokens

```bash
curl -N --no-buffer localhost:3000/ask_stream -H 'content-type: application/json' \
  -d '{"question":"What is the prescription period for obligations?"}'
```

The response is `Transfer-Encoding: chunked` and arrives in 127 separate
chunks. Locally the text is assembled in under 3 ms, so the increments are real
but land faster than a terminal repaints. To see them pace out:

```bash
PRODLLM_STREAM_DEMO_DELAY_MS=60 \
  uv run --group serving bentoml serve prodllm.serving.service:LegalRagService --port 3000
```

Then the same 127 fragments arrive ~60 ms apart over about 8 seconds. That
setting is a demonstration aid: it defaults to 0, every latency figure in
[reports/module-3.md](reports/module-3.md) was measured at 0, and it has no
effect once an LLM is configured — there the pacing is the generator's.

Do not pipe the stream through `head` or `grep` while checking this: the pipe
buffers, and every chunk then appears to arrive at the same instant.

### Building and containerizing the Bento

```bash
UV_INDEX_STRATEGY=unsafe-best-match uv run --group serving bentoml build
uv run --group serving bentoml containerize prodllm-rag:latest
```

The environment variable is required, not optional. BentoML locks dependencies
with `uv pip compile`, and by default uv takes each package from the first
index that offers it at all — the PyTorch CPU index (needed to avoid a ~5 GB
CUDA image) carries an ancient `certifi`, which makes BentoML's own requirement
unsatisfiable. The reasoning and the security tradeoff are documented in
[bentofile.yaml](bentofile.yaml). With it, the lock pins `torch==2.14.1+cpu`
and one `nvidia-*` package instead of fifteen.

## Measured latency

50 concurrent users for 3 minutes against the Bento, **9131 requests, 0
failures**:

| Endpoint | p50 | **p95** | p99 | req/s |
|---|---|---|---|---|
| `POST /ask` | 42 ms | **81 ms** | 210 ms | 27.3 |
| `POST /ask_stream` (time to first byte) | 46 ms | **85 ms** | 200 ms | 6.7 |
| `POST /status` | 6 ms | **13 ms** | 18 ms | 3.4 |

```bash
uv sync --group load
uv run --group load locust -f load/locustfile.py --host http://127.0.0.1:3000 \
  --users 50 --spawn-rate 5 --run-time 3m --headless \
  --html reports/locust/report.html --csv reports/locust/run
```

Report: [reports/locust/report.html](reports/locust/report.html).

`/status` touches no model, so it is the control: its p95 of 13 ms against
`/ask`'s 81 ms shows the service is not queue-bound — the 81 ms is inference
working, not requests waiting. Full analysis, including why streaming TTFB and
total are currently identical, is in
[reports/module-3.md](reports/module-3.md).

## Batch re-indexing

Publishing one new article should not cost a full re-embed of the corpus.

```bash
# offline: extend the index on disk
uv run python scripts/reindex.py --chunks tests/fixtures/new_articles.sample.json --dry-run
uv run python scripts/reindex.py --chunks tests/fixtures/new_articles.sample.json

# live: no restart
curl -s localhost:3000/reindex -H 'content-type: application/json' \
  --data-binary @payload.json      # {"documents": [...], "persist": false}
```

One document into a 1092-document index embeds in **0.11 s**, against minutes
for a full rebuild. The job is **idempotent** — a chunk id already indexed is
reported as `skipped` and not embedded again, so re-running after a partial
failure cannot duplicate it. Duplicates would not merely waste space:
`retrieve` dedupes by article number over a fixed candidate pool, so a doubled
chunk silently consumes a slot a genuine article should have had.

The live path is copy-on-write — it builds an extended index and rebinds it —
because `faiss.IndexFlatIP.add` is not safe against a concurrent `search`, and
a mutex on retrieval would throttle the hot path. It **adds but does not
revise**: amending an existing article still needs `scripts/build_index.py`.

`tests/fixtures/new_articles.sample.json` is a labelled synthetic fixture, not a real
provision. See [reports/module-3.md](reports/module-3.md) for why, and for the
corpus gap (articles 54–68) that re-indexing is the right tool for.

## Canary rollout

```bash
docker compose --profile canary up
```

```
            ┌─────────────── nginx :8080 ───────────────┐
  client ──►│  weight=9  ──────────────►  bento (stable) │
            │  weight=1  ──────────────►  bento-canary   │
            └────────────────────────────────────────────┘
```

Two Bento replicas behind an nginx weighted split. They are generated from one
YAML anchor and differ in **exactly one** environment variable,
`PRODLLM_RELEASE_CHANNEL` — a canary differing in more than the thing being
rolled out cannot attribute a latency change to it.

Only the router publishes a port. During a rollout every request must go
through the split, or you measure whichever replica the caller happened to
address.

**Why 10%.** The point of a canary is to expose a fault to as few callers as
possible while still generating enough traffic to see it. At the ~27 req/s this
service sustains on `/ask`, 10% is still ~2.7 req/s, which fills a p95 bucket
in under a minute.

**Watch these during the rollout**, comparing canary against stable rather than
against an absolute target:

| Signal | Where |
|---|---|
| Which replica served a request | `release_channel` in the `/ask` body, and nginx's `X-Upstream` header |
| Error rate per variant | `X-Upstream` on non-2xx — it survives a 502 from a canary that died before answering |
| p95 per variant | the `release_channel` field on every request log line |
| Readiness | `POST /status` on each replica directly (`documents_indexed`) |

**Promote** by shifting the weights in
[docker/canary-nginx.conf](docker/canary-nginx.conf) — 9/1 → 1/1 → 0/1 — and
reloading nginx. **Roll back** by setting the canary weight to 0 or stopping
the container. Neither step rebuilds an image or redeploys the stable replica.

In a cluster the same shape is two Deployments behind one Service with the
replica counts as the weights, or an Istio `VirtualService` with explicit
weights. The reasoning above is unchanged; what matters is the ratio and what
you measure during it.

One configuration detail is load-bearing: `proxy_buffering off`. With buffering
on, nginx accumulates the whole body and hands the client one block at the end.
That would not break `/ask`, but it silently destroys `/ask_stream` — the
endpoint still returns the right bytes, just never incrementally.

## Serving the generative LLM with vLLM

```bash
docker compose --profile vllm up -d vllm
```

Then point the service at it and restart:

```bash
PRODLLM_LLM_BASE_URL=http://localhost:8000/v1
PRODLLM_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

No application code changes. vLLM serves the same OpenAI-compatible
`/chat/completions` with SSE streaming that `prodllm.streaming` already speaks,
so the adapter cannot tell a self-hosted vLLM from a hosted endpoint — the
backend is a two-variable configuration choice.

The profile pins `Qwen/Qwen2.5-0.5B-Instruct` because the GPU here has 4 GB of
VRAM: at fp16 that fits a 0.5B model with room for a KV cache, and does not fit
a 1.5B one. A 0.5B model's Arabic legal reasoning is weak — a hardware limit,
not a tuning problem — so this profile demonstrates serving infrastructure, not
answer quality.

**vLLM is not in the measured path.** The latency figures above were taken with
no LLM configured, where the service streams retrieved article text verbatim.
That is a deliberate project decision, recorded in
[reports/module-3.md](reports/module-3.md).

## Run from source

```bash
uv sync
uv run uvicorn prodllm.api.main:app --port 8000
```

Requires [uv](https://docs.astral.sh/uv/) 0.12+; it installs Python 3.12 and
every pinned dependency itself. The service needs two artifacts, neither of
which is in git:

**1. The FAISS index** (`artifacts/index/`) — required, or `/ask` returns 503:

```bash
uv run --group corpus python scripts/build_corpus.py   # PDF  -> data/processed/
uv run python scripts/build_index.py                   # JSON -> artifacts/index/
```

`build_corpus.py` needs the source PDF at `data/raw/`. If you already have
`data/processed/chunks.json`, skip straight to `build_index.py`.

**2. The weights** — optional locally. With `artifacts/model/` absent and
`PRODLLM_ALLOW_DOWNLOAD=true` (the default), they are fetched from Hugging Face
on first start and cached. To pin the revision and record a SHA-256:

```bash
uv run python scripts/fetch_weights.py
```

## Container

Built from [docker/Dockerfile](docker/Dockerfile). **Build from the repository
root** — the context must contain `src/`, `pyproject.toml`, `uv.lock` and
`artifacts/index/`:

```bash
docker build -f docker/Dockerfile -t mlops-final-project:0.1.0 .
```

Three targets come out of the one file — one for development, two for
production:

| Target | Weights | Dev tooling | Use |
|---|---|---|---|
| `runtime` (default) | baked in, pinned revision | none | the published image; starts in ~20 s, offline |
| `runtime-thin` | downloaded on start | none | the size comparison in the report; needs network |
| `development` | neither (mount or download) | pytest, ruff, pre-commit, uv | the local edit–test loop |

```bash
docker build -f docker/Dockerfile --target runtime-thin -t mlops-final-project:0.1.0-thin .
docker build -f docker/Dockerfile --target development  -t mlops-final-project:0.1.0-dev .
```

The thin image writes its download to `/app/.cache/huggingface`; mount a volume
there or every restart fetches the model again.

### Development vs runtime

The split is not only about size. A runtime image carrying pytest, ruff and a
writable source tree ships a toolchain and test fixtures into production as
attack surface, and `uv sync --no-editable --no-default-groups` is what keeps
them out. What the `development` target adds:

- the **dev dependency group** (pytest, pytest-cov, ruff, pre-commit) and
  `uv` itself, so `pytest` and `ruff` run inside the container; the runtime
  stages copy a finished `/app/.venv` out of the builder and ship no package
  manager at all,
- an **editable install**, so a bind-mounted `src/` is what runs and
  `uvicorn --reload` restarts on an edit instead of requiring a rebuild,
- `PRODLLM_LOG_LEVEL=DEBUG` and no `HEALTHCHECK` — a dev container is
  routinely started with no index on purpose, and `unhealthy` on a container
  behaving as intended trains you to ignore the column.

Both halves share this one file, and therefore the same Python version, the
same locked resolution and the same `libgomp1` fix — so the separation buys
isolation without letting the dev and production environments drift apart.
**Both run as the same unprivileged user:** a dev container running as root
writes root-owned files into your working tree, and a loop that only works as
root breaks the moment it is promoted to runtime.

### Compose

```bash
docker compose up --build
```

Serves on <http://localhost:8000>. `docker compose ps` reports `healthy` only
once the model and index are both loaded, because the compose healthcheck
mirrors the image's rather than probing for a `200`. The thin variant sits
behind a profile so it does not start by default:

```bash
docker compose --profile thin up --build api-thin     # on port 8001
```

The development container sits behind its own profile, with `src/`, `tests/`
and `scripts/` bind-mounted and `artifacts/` mounted read-only so it reuses
the index already built on the host:

```bash
docker compose --profile dev up dev                   # port 8002, reload on edit
docker compose --profile dev run --rm dev pytest
docker compose --profile dev run --rm dev ruff check .
```

All three can run at once — `api` on 8000, `api-thin` on 8001, `dev` on 8002 —
which is what makes "it works in dev" checkable against the image that ships.

Compose is not ceremony here: the next module adds a tracing backend and
possibly a vector database as sibling services, and this file is where they
attach.

### What the image does and does not contain

- Runs as **non-root** `appuser`, UID 10001 — in every target, development
  included. Verify with
  `docker run --rm mlops-final-project:0.1.0 id -u` (and the same against
  `:0.1.0-dev`). In the runtime images the virtualenv is root-owned and
  `artifacts/` is the only path the service can write; the development image
  gives `appuser` its venv and caches, because `uv sync` has to be able to
  write them.
- On Linux, torch comes from PyTorch's **CPU index**, not PyPI. The PyPI wheel
  pulls 15 `nvidia-*` packages plus triton — about 5 GB of CUDA runtime for a
  model that runs on CPU. This is invisible on Windows, where the PyPI wheel is
  already CPU-only, so it only ever lands in the image.
- `libgomp1` is installed explicitly. `faiss-cpu`'s wheel links
  `libgomp.so.1`, which `python:3.12-slim` does not ship; without it the
  container starts, serves 503 forever, and buries the real `ImportError` in
  the startup log.
- No `src/` and no dev dependencies: `uv sync --no-editable
  --no-default-groups` installs `prodllm` as a real package.
- Secrets are never baked in. `.env` is in `.dockerignore`; pass configuration
  with `--env-file .env` or `-e`.

## Configuration

Every setting is read from the environment with a `PRODLLM_` prefix and has a
working default. [.env.example](.env.example) is the annotated template; the
values that matter most:

| Variable | Default | Notes |
|---|---|---|
| `PRODLLM_ARTIFACTS_DIR` | `artifacts` | **Relative** by default. An absolute default is the "works on my laptop, fails in the container" trap; the image sets `/app/artifacts`. |
| `PRODLLM_ALLOW_DOWNLOAD` | `true` | The default image sets `false`, so a bad artifacts path fails loudly instead of silently downloading. |
| `PRODLLM_SOURCE_MODEL_ID` | `intfloat/multilingual-e5-small` | Changing it invalidates the index — rebuild, or retrieval compares vectors from two different embedding spaces. |
| `PRODLLM_TOP_K` | `5` | Articles cited per answer. |
| `PRODLLM_MIN_SCORE` | `0.0` | Cosine floor, disabled. See the limitation below — this default is a measured result. |
| `PRODLLM_LOG_LEVEL` | `INFO` | `DEBUG` adds a per-call latency line from the `@log_latency` decorator. |
| `PRODLLM_TORCH_NUM_THREADS` | `1` | Scale with replicas, not in-process threads. |
| `PRODLLM_LLM_BASE_URL`, `PRODLLM_LLM_MODEL` | empty | Both empty = extractive answers. Set them to point at a hosted endpoint **or** a self-hosted vLLM — same wire format either way. |
| `PRODLLM_LLM_API_KEY` | empty | A secret. Pass at run time; never commit a value. |
| `PRODLLM_RELEASE_CHANNEL` | `stable` | Which rollout slot this replica is. Returned on `/ask` and `/status`, and on every log line — this is what makes a canary attributable. |
| `PRODLLM_STREAM_DEMO_DELAY_MS` | `0` | Paces the extractive stream so increments are visible in `curl`. A demo aid: keep it at 0 for anything measured. |

Logs are JSON on stdout, one line per request, with a correlation ID that also
comes back as the `x-request-id` header — so a client can quote the header and
you can find the exact request:

```json
{"timestamp": "2026-10-22T15:50:38.185912+00:00", "level": "INFO", "logger": "prodllm.api",
 "message": "Request completed", "correlation_id": "10dbe781-9011-4685-bc33-35230106dcc0",
 "method": "POST", "path": "/ask", "status_code": 200, "latency_ms": 47.52}
```

## Tests and quality gates

```bash
uv run pytest                  # 198 tests, coverage gate at 70% (currently ~97%)
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install && uv run pre-commit install --hook-type pre-push
```

The suite runs in a fresh clone with **no model weights and no index on disk**:
all file and session loading is confined to `from_*` classmethods, so the
tokenize → forward → pool → normalize path is covered with mocks and a
`FakeEmbedder`. Without that seam the most failure-prone module in the
repository would be the least tested.

`pre-commit` runs ruff, formatting, a large-file guard and the `openapi.json`
export on every commit, and the full suite with the coverage gate on every
push. The same checks run in CI, plus an index rebuild and the RAGAS
faithfulness gate — the local hooks stay because a gate that only exists on a
runner is one `--no-verify` and a force-push away from being bypassed.

The suite needs no API key and no network: the faithfulness tests drive a stub
metric, and the MLflow tests skip unless the `tracking` group is installed, so
`uv sync --locked` followed by `uv run pytest` is enough.

## Experiments, data versioning and the quality gate

The corpus is rebuilt from the PDF by a DVC pipeline, chunking configurations
are compared in MLflow, and the winner is promoted to Production in the Model
Registry. Three commands cover the loop:

```bash
docker compose --profile tracking up -d mlflow                    # UI on :5000
uv run --group tracking --group eval python scripts/sweep_chunking.py
uv run --group tracking python scripts/register_best.py
```

Without Docker, `uv run --group tracking python scripts/mlflow_server.py`
starts the same server on the same port.

### The data pipeline

```bash
uv run dvc pull            # fetch Law.pdf and the derived JSON
uv run dvc repro           # PDF -> articles.json + chunks.json -> artifacts/index/
uv run dvc push            # publish what changed
```

Two stages, split along the two real costs: parsing the PDF is seconds and
depends on the extraction code, embedding 2183 vectors is minutes and depends
on the encoder. Editing a parser regex therefore does not re-embed the corpus,
and swapping the encoder does not re-parse the PDF.

[params.yaml](params.yaml) holds `chunk_size`, `overlap` and `embedding_model`,
and both stages declare them as dependencies — so the promoted configuration
and the configuration `dvc repro` builds cannot drift apart. `register_best.py`
writes the winner back into that file.

**The remote is a local directory** (`../.dvc-remote`, relative so a clone
alongside this repository resolves it identically). `dvc push` and `dvc pull`
both work and the mechanism is fully demonstrable, but a GitHub runner cannot
reach it — which is what shapes the CI design below. Moving the remote to S3
or Google Drive is a one-line config change and the only thing standing
between CI and a full rebuild.

### Comparing chunking configurations

`scripts/sweep_chunking.py` runs six configurations over three axes, rebuilding
the index for each — chunk size and overlap change what is *in* the index, so a
sweep that reused one index would compare six labels over identical vectors.
Each run logs `chunk_size`, `overlap`, `embedding_model`, RAGAS faithfulness,
retrieval hit rate and the recorded answers as an artifact.

**The sweep needs a real generator LLM.** With the extractive generator every
answer is a verbatim copy of its contexts, so faithfulness is ~1.0 for every
configuration and the comparison is empty. The script refuses to run that way
rather than producing a flat table that looks like a result. Set
`PRODLLM_LLM_BASE_URL` and `PRODLLM_LLM_MODEL` — see [.env.example](.env.example).

`scripts/judge_sweep.py` then re-judges every recorded run against a single
judge instance, because the judge is not deterministic and six judging sessions
spread over an hour carry drift that is indistinguishable from a difference
between configurations.

### The faithfulness gate

```bash
uv run --group eval python scripts/eval_gate.py
```

Fails below 0.75 RAGAS faithfulness on the fixed twenty-question set in
[eval/questions.jsonl](eval/questions.jsonl). The set is fixed in the
mechanical sense: `load_questions` refuses a set whose size has changed, because
a gate measured against a question set any commit may quietly shrink is not a
gate.

The gate judges [eval/promoted_run.json](eval/promoted_run.json) — the answers
recorded when the current configuration was promoted — rather than generating
fresh ones. That is the local DVC remote again: CI cannot pull the corpus, so
it cannot rebuild the index, so it cannot generate new answers. The gate
therefore catches a promoted configuration whose answers are not supported by
their own contexts, and any tampering with the committed run or the question
set; it cannot catch a retrieval regression introduced after promotion. See
[reports/module-2.md](reports/module-2.md).

## Project layout

```
src/prodllm/
├── rag.py                  corpus -> chunks -> FAISS index; question -> answer + citations
├── corpus/                 PDF extraction: column-aware parse, bidi repair, normalize, validate
├── embedding/              Embedder ABC, FakeEmbedder, TorchEmbedder (transformers)
├── pooling.py              mean pooling + L2 normalization, pure numpy, unit-tested
├── streaming.py            async token streaming; OpenAI-compatible SSE + extractive
├── api/                    routes, schemas, DI seams, lifespan, correlation-ID middleware
├── serving/                BentoML: engine.py holds the behaviour, service.py the decorators
├── instrumentation.py      the @log_latency decorator
├── logging_conf.py         JSON formatter
└── config.py               pydantic-settings, PRODLLM_ prefix
├── eval/                   question set loading, the recorded run, RAGAS faithfulness
└── tracking/               the chunking configuration as a registered MLflow model
scripts/
├── build_corpus.py         the source PDF   -> data/processed/articles.json + chunks.json
├── build_index.py          chunks.json      -> artifacts/index/ (full rebuild)
├── reindex.py              new chunks       -> appended to an existing index
├── fetch_weights.py        Hugging Face     -> artifacts/model/ + metadata.json
├── export_openapi.py       the app          -> openapi.json
├── mlflow_server.py        the tracking server, without Docker
├── sweep_chunking.py       six configurations -> six MLflow runs
├── judge_sweep.py          re-judge every run in one pass
├── register_best.py        promote the winner to Production
└── eval_gate.py            the RAGAS faithfulness gate (exit 1 = fail)
openapi.json                the committed contract, generated, --check in a hook
dvc.yaml, params.yaml       the PDF -> corpus -> index pipeline and its parameters
eval/questions.jsonl        the fixed 20-question test set the gate is defined on
eval/promoted_run.json      the promoted configuration's recorded answers
docker/Dockerfile           multi-stage; development, runtime, runtime-thin
docker/canary-nginx.conf    the 90/10 weighted split used by the canary profile
docker-compose.yml          api :8000, api-thin :8001, dev :8002, mlflow :5000,
                            canary router :8080, vllm :8000 (profiles)
bentofile.yaml              how `bentoml build` packages the serving layer
load/locustfile.py          the 50-user load test, driven by eval/questions.jsonl
tests/fixtures/            sample chunks, and a labelled fixture for re-indexing
reports/module-2.md         the sweep comparison, the registry, and what the gate misses
reports/module-3.md         the serving design, the measured p95, and the canary
reports/locust/             the load-test report and its raw CSVs
```

## Known limitations

These are stated rather than hidden, because each one is a real constraint a
reviewer would otherwise find.

- **No similarity threshold separates relevant from irrelevant.** Measured on
  this corpus over 10 on-topic and 5 off-topic questions, top-1 cosine spans
  0.838–0.899 on-topic and 0.789–0.859 off-topic. The ranges overlap: "Explain
  quantum entanglement" scores 0.859, above several genuine legal questions.
  e5-family models compress similarities into a narrow high band, so
  `min_score` defaults to disabled rather than to a value tuned to look good on
  a handful of examples. The real fix is a cross-encoder reranker, which scores
  query/passage pairs jointly and does separate.
- **Some English article bodies absorb the heading that follows them.**
  Article 388's English text runs on into "Chapter VI Proof of Obligations …".
  Arabic text and all article numbering are unaffected, and no citation points
  at the wrong article, but a few English quotes carry trailing heading text.
- **Lam-alef ligature artifact in the source PDF.** A handful of Arabic words
  extract with `لا` as `ال` (`فلا` → `فال`). This is not repaired, because `ال`
  is also the definite article, so a blanket substitution would corrupt
  thousands of correct words. It is a property of the source PDF's glyph
  encoding.
- **The `topic` field is unpopulated.** Book, chapter and section are captured
  from running headers; the finest hierarchy level is not.
- **No ONNX.** Deferred by decision, not oversight; see
  [reports/module-1.md](reports/module-1.md). DVC and the CI pipeline were
  deferred alongside it and have since landed — see
  [reports/module-2.md](reports/module-2.md).
- **The faithfulness gate re-judges recorded answers, not fresh ones**, and the
  DVC remote is a local directory that CI cannot reach. These are the same
  constraint seen from two ends, and one config change lifts both.
- **vLLM is wired but not in the measured path.** The latency figures were
  taken with no LLM configured, so they measure retrieval and transport rather
  than token generation. Starting the `vllm` profile and setting two
  environment variables puts a generator in the path, at which point those
  numbers need re-measuring. A deliberate decision, recorded in
  [reports/module-3.md](reports/module-3.md).
- **Batch re-indexing adds but does not revise.** A flat FAISS index can delete
  by id, but doing so compacts and renumbers the remaining rows, which would
  desynchronise the metadata the index positions resolve through. Amending an
  existing article still needs a full rebuild.
- **`dvc.lock` has no `build_index` stage.** `dvc.yaml` defines it but only
  `build_corpus` is locked, so the index currently has no recorded hash to
  verify against and `dvc status` reports it as changed. Predates Module 3.
- **A correct refusal scores zero faithfulness.** When retrieval misses, the
  generator says so instead of inventing a provision — the right behaviour —
  and RAGAS scores that refusal 0.0. One question in the fixed set does this
  on every configuration, which puts a ceiling of roughly 0.95 on the metric
  and means faithfulness here partly proxies retrieval quality.
