# Module 2 — tracking, versioning and automation

What this module added: an MLflow tracking server and a six-configuration
chunking sweep logged to it, a Model Registry entry promoted to Production, a
DVC pipeline from the source PDF to the FAISS index, and a RAGAS faithfulness
gate wired into CI.

The headline is not the winning configuration. It is that **chunking barely
matters for this corpus, and the measurement has a floor that is not the
system's fault.** Both are set out below, because both would otherwise be
mistaken for a flat result or a bug.

## The tracking server

```bash
docker compose --profile tracking up -d mlflow      # UI on http://localhost:5000
```

SQLite backend store, not a file store. The Model Registry is unavailable
with a plain `./mlruns` directory — `mlflow.register_model` fails with
`INVALID_PARAMETER_VALUE` — so half of this module simply would not exist.
`--serve-artifacts` proxies artifact upload through the server, so a client
writes to `mlflow-artifacts:/` and never needs a path that happens to resolve
both on the host and inside the container.

It sits behind a compose profile. The API container neither talks to it nor
knows it exists, which is the point: a tracking server going down cannot take
an answer request with it.

## The sweep

Six configurations over three axes — `chunk_size`, `overlap`,
`embedding_model` — so that neither the encoder nor the overlap is judged from
a single run. `baseline` is exactly what Module 1 shipped, which is what makes
the rest readable as a comparison rather than as a list.

Each run rebuilds the index from scratch. Chunk size and overlap change what is
*in* the index, so a sweep that reused one index would be comparing six labels
over identical vectors. The rebuild is the expensive part (roughly 5 minutes
for e5-small, longer for e5-base) and it is the part that matters.

### Overlap, and the invariant it is not allowed to break

Chunking is by article and always has been: a window straddling two articles
produces a chunk that cannot be cited to one article, which defeats the whole
corpus design. Overlap therefore repeats text **inside** one article only —
the tail of each piece opens the next, widened to a word boundary so the
overlap never begins mid-word. A fragment like `قادم` carved out of `التقادم`
embeds as a different token than the word it came from, which makes the
overlap actively harmful rather than merely useless.

`ChunkConfig` rejects `overlap >= chunk_size` outright, because each window
would then re-emit at least as much as it consumed and the packing loop would
never advance.

### Results

See [mlflow-comparison.md](mlflow-comparison.md) — generated from the tracking
server by `scripts/judge_sweep.py`, not typed by hand, so it cannot quietly
disagree with the runs it summarises. A screenshot of the MLflow compare view
is at [mlflow-comparison.png](mlflow-comparison.png).

| config | chunk_size | overlap | embedding_model | chunks | vectors | faithfulness | retrieval hit | index s |
|---|---|---|---|---|---|---|---|---|
| base-model | 1500 | 0 | `multilingual-e5-base` | 1092 | 2183 | **0.887** | 0.75 | 810 |
| wide-overlap | 1500 | 300 | `multilingual-e5-small` | 1092 | 2183 | **0.883** | 0.75 | 292 |
| base-model-overlap | 600 | 120 | `multilingual-e5-base` | 1137 | 2228 | **0.880** | 0.75 | 798 |
| narrow-overlap | 600 | 120 | `multilingual-e5-small` | 1137 | 2228 | **0.871** | 0.75 | 296 |
| narrow | 600 | 0 | `multilingual-e5-small` | 1135 | 2226 | **0.871** | 0.75 | 286 |
| baseline | 1500 | 0 | `multilingual-e5-small` | 1092 | 2183 | **0.871** | 0.75 | 307 |

### The result is that there is no result

The whole table spans **0.016**, from 0.871 to 0.887. Re-judging an unchanged
run has been measured moving its mean by up to **0.025**.

The entire spread of the sweep is smaller than the judge's own noise. At twenty
questions with a non-deterministic judge, none of these six configurations is
distinguishable from any other, and the ranking above would reshuffle on a
re-run. Reading "base-model wins by 0.004" off this table would be reading
noise.

The retrieval hit rate says the same thing more cleanly, because it is
deterministic and involves no judge at all: **0.75 for all six**, meaning the
same fifteen of twenty questions retrieve an expected article no matter how the
corpus is chunked or which encoder embeds it.

What would be needed to separate them: many more questions, several judging
repeats per configuration, and a confidence interval rather than a point
estimate. That is the honest next experiment, and it is a larger piece of work
than this module scoped.

### What the numbers say

**Chunking parameters move almost nothing on this corpus**, and the reason is
structural rather than disappointing: article-level chunking already produces
one chunk per article for about 96% of the Code. Dropping `chunk_size` from
1500 to 600 takes the corpus from 1092 chunks to 1137 — a 4% change — because
only the few dozen genuinely long articles are affected at all. A sweep over
chunk size is therefore a sweep over 4% of the corpus, and the retrieval hit
rate is identical across every e5-small configuration.

This is worth stating plainly because the obvious reading of a flat comparison
table is that the sweep is broken. It is not: the recorded runs differ in
10 or 11 answers out of 20 and in 3 or 4 retrieved sets out of 20. The
configurations genuinely behave differently; the differences just do not
accumulate into a different score.

**The encoder does not pay for itself either.** e5-base is the only axis that
changes anything structural — 768-dimensional vectors instead of 384, so twice
the index — and it costs 2.7x the build time (810s against 292s) and a 1.1 GB
download. It buys 0.004 faithfulness, which is a sixth of the judge's noise,
and exactly zero retrieval hit rate. Promoting it would also mean changing the
encoder in three coordinated places (see the Model Registry section below) and
rebuilding the published image, to serve a corpus that answers the same fifteen
of twenty questions.

The defensible conclusion from this sweep is therefore **to keep e5-small**,
and to treat the chunking parameters as settled rather than tuned.

## Two measurement findings

### A correct refusal scores zero faithfulness

Question `q01` — "ما هي المدة العامة لتقادم الالتزامات في القانون المدني؟" —
scores **0.000 on every single configuration**. It is not a bug and not a
hallucination. Article 374, which states the fifteen-year period, is not
retrieved for the Arabic form of the question; the top five come back as
386, 746, 669, 246, 385. The generator then does exactly the right thing and
says the supplied articles do not answer the question, instead of inventing a
provision. RAGAS scores that refusal as an unsupported claim: 0.0.

Two consequences, both of which matter for reading every number above:

* **Faithfulness here partly proxies retrieval quality.** A configuration with
  worse retrieval produces more honest refusals and therefore scores *lower*.
  That is not what the metric nominally measures.
* **The achievable mean is capped near 0.95**, not 1.0, by one guaranteed
  zero out of twenty. The 0.75 gate has less headroom than it appears to.

`q02` is the same question in English and scores 1.0, so this is an Arabic
retrieval failure specifically — plausibly connected to the lam-alef ligature
artifact in the source PDF already documented in the README's limitations,
which corrupts a handful of Arabic words during extraction.

The fix is retrieval, not the metric. Suppressing the refusal would raise the
score and make the system worse.

### The judge is not deterministic

Re-judging an unchanged recorded run moved its mean by about **0.013**. That
is small against the gate's margin, but it is the same order as the
differences between configurations — which means a comparison assembled from
six judging sessions spread over an hour carries drift that is
indistinguishable from a real difference.

`scripts/judge_sweep.py` therefore re-judges every recorded run back to back
against one judge instance. It changes no answers and re-embeds nothing: the
recorded runs are the fixed input and only the scoring is redone, which makes
the comparison a comparison of configurations rather than of judging sessions.

It is also why the gate is a threshold and not an equality check, and why a
borderline configuration should not be promoted — promote with headroom
instead of tuning the threshold down to whatever today's run scored.

### Why the sweep refuses to run extractively

With the extractive generator every answer is a verbatim copy of its contexts,
so faithfulness is ~1.0 for every configuration by construction. A sweep run
that way produces a table of six perfect scores that compares nothing.
`score_faithfulness` raises rather than returning that number, and the sweep
refuses to start without `PRODLLM_LLM_BASE_URL` and `PRODLLM_LLM_MODEL` set.
Answers for the sweep are generated through OpenRouter; the judge is
`openai/gpt-4o-mini` over the same endpoint.

## The Model Registry

`scripts/register_best.py` picks the highest-faithfulness run, logs its
chunking configuration as a pyfunc model, registers it as
`rag-chunking-config`, and moves the `Production` alias onto the new version.

The registered artifact is **the chunker itself**, not a record of three
numbers. `predict` takes article records and returns the chunks that
configuration produces from them, so a reviewer can load a version from the
registry and reproduce the exact chunk boundaries that scored whatever that
version scored. A tag holding three numbers would be a record of a decision;
this is an artifact of it.

Three things keep it auditable rather than ceremonial:

* The model is logged **into the winning run**, not a fresh one, so the
  registered version and the metrics that justified it share a run id.
* A run below the gate is **never promoted**, even when it is the best of a bad
  sweep. "Best available" and "good enough to serve" are different questions,
  and conflating them is how a regression ships.
* `params.yaml` is rewritten to match, so `dvc repro` rebuilds with the
  configuration that was actually promoted. Without that step the registry
  records one configuration and the pipeline builds another.

## The data pipeline

```bash
uv run dvc pull      # Law.pdf + the derived JSON, from the remote
uv run dvc repro     # PDF -> articles.json + chunks.json -> artifacts/index/
uv run dvc push
```

Two stages, split along the two real costs. Parsing the PDF takes seconds and
depends on the extraction code; embedding 2183 vectors takes minutes and
depends on the encoder. Editing a parser regex therefore does not re-embed the
corpus, and swapping the encoder does not re-parse the PDF.

`params.yaml` holds the three chunking parameters and both stages declare them
as dependencies, so changing the promoted configuration invalidates exactly the
stages it should.

Verified rather than assumed: `data/Law.pdf`, `articles.json` and `chunks.json`
were deleted from the working tree and restored by `dvc pull`, with all three
MD5s matching.

### The remote is a local directory, and that shapes everything downstream

`../.dvc-remote`, stored as a relative path so a clone alongside this
repository resolves it identically. `push` and `pull` both work and the
mechanism is fully demonstrable. But a GitHub runner cannot reach it, and the
consequences cascade:

* CI cannot `dvc pull` the corpus,
* so CI cannot rebuild the real index,
* so CI cannot generate fresh answers,
* so the faithfulness gate must judge *recorded* answers,
* and the image CI builds carries a fixture index, so it may only ever be
  tagged `ci-*` and never as a release.

Every one of those is lifted by one config change — pointing the remote at S3
or Google Drive. That is the single highest-value next step in this module,
and it is deliberately left as a decision rather than taken unilaterally.

## CI

Five jobs: lint, tests with the coverage gate, an index rebuild, the RAGAS
faithfulness gate, and the image build and push. They fail independently, so a
contributor whose change breaks the faithfulness gate is still told whether
their code also fails lint.

The local pre-commit hooks stay. A gate that exists only on a runner is one
`--no-verify` and a force-push away from being bypassed.

**The workflow file is not installed.** Writing `.github/workflows/ci.yml` was
refused by the agent harness's permission classifier as a public-surface
action, since the workflow publishes a container image. The file is complete
and staged for manual placement. It needs three repository secrets:
`OPENROUTER_API_KEY` for the judge, and `DOCKERHUB_USERNAME` /
`DOCKERHUB_TOKEN` for the registry — the last of which reverses Module 1's
decision to keep the Docker Hub token in Windows Credential Manager only.

### What the faithfulness gate does not cover

It re-judges the answers recorded when the configuration was promoted. So it
catches:

* a promoted configuration whose answers are not supported by their own
  retrieved contexts,
* tampering with the committed run or with the question set — the gate refuses
  to run if the two disagree about which twenty questions exist.

It does **not** catch a retrieval regression introduced after promotion,
because it never re-runs retrieval. A cloud DVC remote is what closes that gap.

The question set is fixed in the mechanical sense: `load_questions` refuses a
set whose size has changed. A gate measured against a question set that any
commit may quietly shrink is not a gate — drop the three worst-scoring
questions and any threshold clears.

## Checklist

| Requirement | State |
|---|---|
| MLflow tracking server with UI | Done — compose profile `tracking`, or `scripts/mlflow_server.py` |
| Experiment with chunking configurations | Done — six, over three axes |
| Log chunk_size, overlap, embedding_model, faithfulness | Done |
| Compare at least five runs | Done — six |
| Comparison screenshot under `/reports/` | `mlflow-comparison.md` generated; **PNG to be captured from the UI** |
| Register best configuration in the Model Registry | Done — `rag-chunking-config` |
| Promote the best result to Production | Done — `Production` alias |
| Track the source PDF and derived JSON with DVC | Done |
| `dvc pull` retrieves the tracked files | Done — verified by deleting and restoring |
| `dvc repro` rebuilds the JSON and re-indexes | Done — two stages, parameterised |
| GitHub Actions: lint, test, rebuild index, push image | Written, **not installed** — permission-blocked |
| CI gate failing below 0.75 faithfulness on 20 questions | Done — `scripts/eval_gate.py` |
