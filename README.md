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

## API

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
| `PRODLLM_LLM_BASE_URL`, `PRODLLM_LLM_MODEL` | empty | Both empty = extractive answers. |
| `PRODLLM_LLM_API_KEY` | empty | A secret. Pass at run time; never commit a value. |

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

Gates are local by design — there is no CI pipeline. `pre-commit` runs ruff,
formatting, a large-file guard and the `openapi.json` export on every commit,
and the full suite with the coverage gate on every push.

## Project layout

```
src/prodllm/
├── rag.py                  corpus -> chunks -> FAISS index; question -> answer + citations
├── corpus/                 PDF extraction: column-aware parse, bidi repair, normalize, validate
├── embedding/              Embedder ABC, FakeEmbedder, TorchEmbedder (transformers)
├── pooling.py              mean pooling + L2 normalization, pure numpy, unit-tested
├── api/                    routes, schemas, DI seams, lifespan, correlation-ID middleware
├── instrumentation.py      the @log_latency decorator
├── logging_conf.py         JSON formatter
└── config.py               pydantic-settings, PRODLLM_ prefix
scripts/
├── build_corpus.py         the source PDF   -> data/processed/articles.json + chunks.json
├── build_index.py          chunks.json      -> artifacts/index/
├── fetch_weights.py        Hugging Face     -> artifacts/model/ + metadata.json
└── export_openapi.py       the app          -> openapi.json
openapi.json                the committed contract, generated, --check in a hook
docker/Dockerfile           multi-stage; development, runtime, runtime-thin
docker-compose.yml          api :8000, api-thin :8001 (thin), dev :8002 (dev)
reports/module-1.md         measurements and the maturity self-assessment
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
- **No ONNX, no CI pipeline, no DVC.** All three were deferred by decision, not
  oversight; see [reports/module-1.md](reports/module-1.md).
