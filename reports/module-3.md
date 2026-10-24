# Module 3 — inference and serving

The serving surface for the Egyptian Civil Code Q&A system: a BentoML service
wrapping the existing RAG pipeline, an asynchronous `/ask`, token streaming,
batch re-indexing against a live index, and a measured load test.

Everything below was run on this machine. Where a number is quoted, the command
that produced it is quoted with it.

| | |
|---|---|
| Serving framework | BentoML 1.4.40 |
| Load generator | Locust 2.47.0 |
| Host | Windows 10, 16 logical cores, 15.5 GB RAM available to Docker |
| Encoder | `intfloat/multilingual-e5-small`, 384-dim, safetensors, CPU |
| Index | flat FAISS `IndexFlatIP`, 1092 documents / 2183 vectors |
| Generator in the measured path | extractive (no LLM configured) |

---

## The shape of it

BentoML sits **beside** the FastAPI app from Module 1, not on top of it. Both
import the same `prodllm.rag.RagEngine`.

```
prodllm/
  rag.py              retrieval, citation, refusal      <- shared, one definition
  streaming.py        async token streaming             <- new
  api/                FastAPI: /predict, /metadata, /health, /ask
  serving/
    engine.py         RagServiceCore: all the behaviour <- new
    service.py        BentoML decorators only           <- new
```

Two reasons for that split rather than a port.

The FastAPI app is a Module 1 deliverable with a committed `openapi.json`, a
test suite, and a Module 2 CI gate hanging off it. Replacing it would have
meant rewriting all three to deliver a Module 3 feature.

More usefully: `serving/engine.py` imports no `bentoml`. All of the behaviour —
degraded startup, the async answer path, streaming, re-indexing, release
identity — is a plain class, so it is tested with no server, no port and no
BentoML runtime. `service.py` is 49 statements of request parsing and status
codes. That is what keeps the serving tests as fast and as offline as the rest
of the suite.

### Endpoints

| Endpoint | Why |
|---|---|
| `POST /ask` | Buffered. One JSON object: answer, citations, latency, release channel. |
| `POST /ask_stream` | The same answer, streamed as it is produced. |
| `POST /reindex` | Batch re-indexing against the live index, no restart. |
| `POST /status` | Documents indexed and which channel answered. |
| `GET /readyz` | BentoML's own probe. |

`/status` is deliberately not called `/healthz`. BentoML already serves
`/healthz`, `/livez` and `/readyz`; taking one of those names leaves a path
where `GET` reaches BentoML's probe and `POST` reaches application code — two
different answers at one URL. That was caught by actually curling it, and
`tests/test_serving_service.py` now asserts the application routes stay
disjoint from BentoML's.

---

## Measured latency

```
uv run --group load locust -f load/locustfile.py --host http://127.0.0.1:3000 \
  --users 50 --spawn-rate 5 --run-time 3m --headless \
  --html reports/locust/report.html --csv reports/locust/run
```

50 concurrent users, 5/s ramp, 3 minutes, `PRODLLM_STREAM_DEMO_DELAY_MS=0`.
Report: [`reports/locust/report.html`](locust/report.html), raw CSV in
[`reports/locust/`](locust/).

**9131 requests, 0 failures.**

| Endpoint | reqs | p50 | **p95** | p99 | max | req/s |
|---|---|---|---|---|---|---|
| `POST /ask` | 4902 | 42 ms | **81 ms** | 210 ms | 452 ms | 27.3 |
| `POST /ask_stream` — time to first byte | 1207 | 46 ms | **85 ms** | 200 ms | 420 ms | 6.7 |
| `POST /ask_stream` — total | 1207 | 47 ms | **85 ms** | 200 ms | 420 ms | 6.7 |
| `POST /status` | 608 | 6 ms | **13 ms** | 18 ms | 230 ms | 3.4 |
| Aggregated | 9131 | 42 ms | **81 ms** | 200 ms | 452 ms | 50.9 |

### The headline number

**p95 on `/ask` is 81 ms at 50 concurrent users, with no failed requests.**

The project rubric does not set a universal p95 threshold for this track, so
this is the measured figure rather than a pass against a target.

### What the numbers actually say

**The service is not queue-bound.** `/status` touches no model, so its latency
is the service's own overhead: p95 13 ms against `/ask`'s 81 ms. If requests
were backing up in the admission queue, both would have climbed together.
They did not, so the 81 ms is inference doing work, not waiting. That
comparison is the reason `/status` is in the load mix at all — it is the
control, and no amount of staring at the `/ask` number alone could settle the
question.

**The tail is warm-up, not saturation.** p99 is 210 ms against a p50 of 42 ms.
The first `/ask` after a cold start measured **159 ms**, and a single request
against a freshly started service took **277 ms** to retrieve. Once torch has
done a forward pass and the allocator has settled, the same work is ~42 ms. The
long tail is concentrated early in the run, during the ramp.

**Streaming TTFB ≈ streaming total, and that is an artifact of the generator.**
85 ms for both. With no LLM configured the answer text is assembled from
retrieved articles in well under a millisecond, so there is nothing to spread
out: essentially all of the 85 ms is retrieval, and the fragments then flush
back to back. These two columns would diverge sharply with a real generator,
where total time is dominated by token production and TTFB is not. **So this
run measures retrieval and transport under load — it does not characterise
token-generation latency**, and no figure here should be read as doing so.

---

## Streaming

`/ask_stream` returns `Transfer-Encoding: chunked` with no `Content-Length`,
and the body arrives in **127 separate network chunks**:

```
chunks received      : 127
first chunk at       : 277.2 ms
last chunk at        : 280.0 ms
spread first..last   : 2.8 ms
VERDICT: incremental -- chunks arrived spread over time
```

127 chunks rather than one proves the transport is genuinely incremental. But a
2.8 ms spread is not *visibly* incremental — in `curl` it looks like one block,
because locally the extractive text is produced faster than a terminal
repaints. Setting `PRODLLM_STREAM_DEMO_DELAY_MS=60` paces the same 127
fragments:

```
chunks received      : 127
first chunk at       : 269.3 ms
last chunk at        : 8024.3 ms
spread first..last   : 7755.0 ms

first arrivals (ms):  269.3  328.3  383.7  450.6  516.8  582.4 ...
```

~60 ms apart, exactly as configured, and unmistakably progressive in `curl -N`.

That delay is a demonstration aid and nothing else. It defaults to 0, every
number in this report was taken at 0, `tests/test_config.py` asserts the
default is 0, and it has no effect once an LLM is configured — there the
pacing is the generator's.

One measurement trap worth recording: piping the stream through `head -c` makes
every chunk appear to arrive at the same instant, because the pipe buffers.
The timings above come from a client reading the socket directly.

### Retrieval settles before any token is emitted

`RagServiceCore.astream` awaits retrieval in the threadpool, then streams. The
ordering is not incidental: a caller cannot verify a claim against an article
that has not been chosen yet, so emitting prose first would make the stream
look grounded before it was. It also means TTFB on this endpoint has a floor of
one retrieval, which the numbers above show is most of it.

---

## Asynchronous, and why it had to be

The handlers are `async def`, but the work inside them — a torch forward pass
and a FAISS search — is blocking CPU, not I/O. It is handed to a threadpool
with `run_in_threadpool`. Awaiting it directly on the event loop would stall
every other in-flight request, and at 50 concurrent users that is the
difference between an 81 ms p95 and a service that serialises.

`tests/test_serving_core.py::test_answer_is_actually_concurrent` gathers two
answers to pin the property down.

The replica advertises `concurrency: 64`, above the 50 the load test uses. That
is deliberate: with a lower limit the run would have measured BentoML's
admission throttle rather than this service's latency. A test asserts it stays
≥ 50.

---

## Batch re-indexing

Verified end to end, with the real encoder, in both languages.

Adding one document to a 1092-document index:

```
$ uv run python scripts/reindex.py --chunks tests/fixtures/new_articles.sample.json --json
  documents indexed : 1092
  new               : 1
embedding 1 new documents ...
  embedded in 0.11s
  added: 1   skipped: 0   vectors added: 2   documents indexed: 1093
```

**0.11 s to embed one document**, against minutes to re-embed the corpus. That
is the whole argument for an incremental path: publishing an amendment should
cost roughly the time to embed the amendment, and a full rebuild also rewrites
2183 vectors that nothing asked to change.

`added: 1` but `vectors added: 2`, because a bilingual chunk contributes one
vector per language. Conflating those two counts is how "documents indexed"
ends up reported at twice its real value.

### Live, with no restart

```
$ curl -s -X POST localhost:3000/status   -d '{}'
{"documents_indexed": 1092, "release_channel": "stable", ...}

$ curl -s -X POST localhost:3000/reindex  --data-binary @payload.json
{"added": 1, "skipped": 0, "vectors_added": 2, "documents_indexed": 1093}

$ curl -s -X POST localhost:3000/ask \
    -d '{"question":"synthetic fixture record used to verify batch re-indexing end to end"}'
  art  9001  score 0.8876   <-- added moments earlier
  art   444  score 0.8549
  art   849  score 0.8013
```

Added live, retrievable immediately, ranked first. The Arabic query for the
same document also ranks it first (0.8998).

### Idempotent

Re-posting the same batch reports `{"added": 0, "skipped": 1}`. Not a
convenience: `retrieve` dedupes by article number over a fixed candidate pool,
so a duplicated chunk would silently consume a slot and push a genuine article
out of the top-k. Re-running the job after a partial failure, or pointing it at
the whole corpus file, is therefore safe.

### Copy-on-write, so retrieval needs no lock

`faiss.IndexFlatIP.add` is not safe against a concurrent `search`. Mutating the
served index in place would mean either a race or a mutex on every retrieval —
and retrieval is the hot path the load test measures.

So the live path builds an extended copy (`faiss.clone_index`, a few MB at this
scale) and rebinds one attribute. A request already inside `retrieve` finishes
against a consistent snapshot; the next one sees the new rows. The offline
script appends in place instead, because nothing is reading it.

### What the fixture is, and is not

`tests/fixtures/new_articles.sample.json` is a **synthetic fixture**, labelled as such in
every field. It is not a provision of the Egyptian Civil Code and must not be
read as one.

That choice is worth stating. The parsed corpus is missing 57 article numbers,
including the contiguous range **54–68** — those articles exist in
`data/Law.pdf` but the extractor did not produce them. A genuinely missing
article would have been the ideal re-indexing subject, and batch re-indexing is
exactly the right tool for backfilling them once the parser is fixed. But
reconstructing their text is a corpus-parsing problem (Module 0's), and
inventing legal text to fill the gap would put fabricated provisions in a
system whose entire purpose is answering questions about real law. So the
mechanism is demonstrated with an unmistakable fixture, and the corpus gap is
recorded here as outstanding work rather than papered over.

### Known limitation: it adds, it does not revise

A flat index can delete by id, but doing so compacts and renumbers the
remaining rows, which would desynchronise the metadata list that index
positions resolve through. Amending an existing article therefore still needs a
full rebuild via `scripts/build_index.py`. Appending is exact and costs no
recall; revising is not supported.

---

## vLLM: wired, documented, and not in the measured path

**This is the one checklist item not met as specified, and it was a deliberate
decision.** The original plan was a local vLLM server on this machine's GPU.
The project owner chose to keep the hosted OpenAI-compatible endpoint
(OpenRouter) as the generator instead, knowing the tradeoff.

What that leaves:

- vLLM is **not** in the request path, so the 81 ms p95 above is not a
  vLLM measurement and is not presented as one.
- vLLM **is** wired up and documented: `docker compose --profile vllm up -d
  vllm` starts it, and `PRODLLM_LLM_BASE_URL` / `PRODLLM_LLM_MODEL` point the
  service at it. No application code changes, because vLLM serves the same
  OpenAI-compatible `/chat/completions` with SSE streaming that
  `prodllm.streaming` already speaks — the adapter cannot tell the two apart.

The hardware constraint behind the original recommendation is real and worth
recording. The GPU here is an **RTX A2000 Laptop with 4 GB of VRAM** (verified
to pass through to Linux containers — `docker run --gpus all` reports it). At
fp16 that fits a 0.5B model (~1 GB of weights, leaving room for a KV cache);
a 1.5B model does not. So the compose profile pins
`Qwen/Qwen2.5-0.5B-Instruct` at `--gpu-memory-utilization 0.85`. A 0.5B
model's Arabic legal reasoning is weak — that is a hardware limitation, not a
tuning problem, and it is why serving infrastructure rather than answer quality
is what that profile demonstrates.

### No generator is configured at all right now

`PRODLLM_LLM_BASE_URL` and `PRODLLM_LLM_MODEL` are empty in `.env`, and no API
key is set anywhere in the repository or the environment. The service therefore
falls back to the extractive generator, which quotes retrieved articles
verbatim with their citations.

That fallback is a documented design choice, not a failure mode: verbatim
article text cannot hallucinate, which is the one thing this track treats as
unacceptable. It is also what made the load test meaningful. Driving 9131
requests through a hosted endpoint at 50 concurrent users would have cost real
money, hit rate limits, and measured someone else's network — the latency of
the thing actually built here is better measured without it.

**To re-measure with a real generator**, set `PRODLLM_LLM_API_KEY`,
`PRODLLM_LLM_BASE_URL` and `PRODLLM_LLM_MODEL`, restart, and re-run the Locust
command above. Expect the two `/ask_stream` columns to diverge: TTFB should
stay in the same neighbourhood (it is retrieval-bound), while total time grows
with the answer length.

---

## Canary rollout

Configured as a runnable topology, not just prose. `docker compose --profile
canary up` brings up two Bento replicas and an nginx weighted splitter.

```
            ┌─────────────── nginx :8080 ───────────────┐
            │  weight=9  ──────────────►  bento (stable) │
            │  weight=1  ──────────────►  bento-canary   │
            └────────────────────────────────────────────┘
```

The replicas are generated from one YAML anchor and differ in **exactly one
environment variable**, `PRODLLM_RELEASE_CHANNEL`. That is verified rather than
asserted:

```
env keys differing: ['PRODLLM_RELEASE_CHANNEL']
```

A canary that differs in more than the thing being rolled out cannot attribute
a latency change to it.

The full configuration and the reasoning for each choice — why 10% rather than
50%, why `proxy_buffering off` is load-bearing for `/ask_stream`, why the proxy
timeouts sit just above the backend's — is in
[`docker/canary-nginx.conf`](../docker/canary-nginx.conf), and the operational
procedure is in the README.

### What makes it observable

`release_channel` is returned in every `/ask` body, reported by `/status`, and
attached to every request log line. During a rollout, "which variant served
this?" is the first question asked, and without it the answer is a guess.

nginx additionally sets `X-Upstream`. That is not redundant: the response body
only helps for requests that reached application code, whereas the header is
set by the proxy and survives a 502 from a canary that died before answering —
which is precisely the request most worth attributing.

---

## Two notes on the repository

**`dvc.lock` has no `build_index` stage.** `dvc.yaml` defines it, but only
`build_corpus` is locked, so `dvc status` reports `artifacts/index` as a
changed output. This predates Module 3 and was not introduced here; it is
flagged because it means the index currently has no recorded hash to verify
against. Left alone deliberately rather than silently committing data-versioning
state as a side effect of a serving change.

**`build_index` depends on `src/prodllm/rag.py`.** Correct — pooling and
vectorisation live there. But `rag.py` now also holds the re-indexing code, so
editing re-indexing logic invalidates the index stage and triggers a full
re-embed that nothing semantically requires. Worth splitting if that path is
touched often.

---

## Checklist

| Deliverable | Status |
|---|---|
| BentoML wraps the RAG pipeline | Done — `prodllm.serving`, 4 endpoints, `bentoml build` produces a locked Bento |
| vLLM serves the generative LLM | **Not in the measured path** — owner chose the hosted endpoint; vLLM wired behind a compose profile and documented |
| Asynchronous `/ask` | Done — `async def` + threadpool offload, concurrency asserted by test |
| Tokens stream progressively in curl | Done — 127 chunks, chunked encoding; visibly paced via the demo delay |
| Locust with 50 concurrent users | Done — 9131 requests, 0 failures, 3 min |
| Locust report under `/reports/` | Done — `reports/locust/report.html` + CSVs |
| p95 latency documented | Done — **81 ms** on `/ask` |
| Canary rollout documented in the README | Done — runnable `canary` profile, 90/10 split, promotion and rollback steps |
| Batch re-indexing implemented and tested | Done — script + live endpoint, idempotent, copy-on-write, new document verified retrievable |

Test suite: **356 tests, 96.8% coverage**, no network and no model weights
required.
