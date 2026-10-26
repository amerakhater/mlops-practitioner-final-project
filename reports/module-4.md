# Module 4 — optimization

Two optimizations, and they pull in opposite directions. A cross-encoder
re-ranker is added to retrieval, which **costs** latency and buys accuracy. The
generative model is quantized to 4 bits, which **buys** memory and speed and
costs a little accuracy. Module 4 is about measuring both trades rather than
asserting either.

The short version:

| | before | after |
|---|---|---|
| Retrieval MRR | 0.597 | *see [The re-ranker](#the-re-ranker)* |
| Retrieval hit@1 | 0.500 | *see below* |
| Generator weights on disk | 2 944 MiB (BF16) | **1 097 MiB (4-bit GPTQ) — 2.69×** |
| Generator perplexity, held-out corpus | 15.355 | **16.270 (+5.96 %)** |
| Qwen2.5-1.5B on this 4 GiB GPU | does not fit (4.48 GiB needed) | fits (2.37 GiB) |

## The hardware, first, because it decided everything

Every number below was taken on one machine, captured programmatically by
`prodllm.bench.hardware` rather than typed into this file:

```
Intel Core (Tiger Lake, family 6 model 141), 16 logical cores, 31.7 GiB RAM
GPU: NVIDIA RTX A2000 Laptop GPU, 4096 MiB VRAM, driver 591.66
torch 2.14.1+cpu  ->  torch.cuda.is_available() == False
```

Read those last two lines together, because their combination is the single
fact that shaped this module. There **is** a CUDA device in this machine, and
the installed PyTorch **cannot use it**: the Windows wheel on PyPI is CPU-only,
and this project pins it that way on purpose — `pyproject.toml` notes that the
Linux CUDA build drags in fifteen `nvidia-*` wheels and roughly 5 GB for a
model that runs on CPU.

So `prodllm.bench.hardware.Hardware` records "GPU present" and "GPU reachable
from torch" as two separate fields. Collapsing them into one `has_gpu` boolean
is how a report ends up claiming a measurement the interpreter could not have
made, and this module is full of opportunities to do exactly that.

Three consequences, stated once and relied on throughout:

1. **Everything here runs on CPU.** Where a number is a latency, it is a CPU
   latency and is not comparable with a GPU figure from a model card.
2. **4-bit *inference* kernels cannot be benchmarked here.** They are CUDA
   kernels. Where this report gives a 4-bit quality number it says how it was
   obtained, and where it cannot give a 4-bit speed number it says so instead
   of substituting an estimate.
3. **The VRAM arithmetic is arithmetic, not measurement**, and is presented as
   such — derived in `scripts/vram_table.py` from architecture dimensions so a
   reader can check it.

## The re-ranker

### Why there is one, and why it is this module's job

The first stage is a FAISS `IndexFlatIP` over 2 183 unit-norm vectors. It
scores a query against each article *independently* — the query vector and the
passage vector are computed without ever seeing one another — which is what
makes it fast enough to sweep the whole corpus per request, and also what
limits it.

`prodllm.rag.DEFAULT_MIN_SCORE` has carried the measurement since Module 1: on
ten on-topic legal questions against five clearly off-topic ones,

```
on-topic  top-1 cosine: min 0.838  max 0.899  mean 0.870
off-topic top-1 cosine: min 0.789  max 0.859  mean 0.814
```

The ranges **overlap**. "Explain quantum entanglement" scores 0.859, above
several genuine legal questions. e5-family encoders compress similarities into
a narrow high band, so no absolute threshold separates relevant from
irrelevant, and `min_score` has therefore shipped at `0.0` — disabled — for
three modules. The comment next to it has said all along that the real fix is a
cross-encoder over the top-k, "deliberately left to a later module rather than
faked with a threshold." This is that module.

### The baseline, and the ceiling

Measured by `scripts/eval_retrieval.py` over the fixed twenty-question set,
against the real index, single-stage:

```
retrieval_hit_rate     0.7500
retrieval_mrr          0.5975
retrieval_hit_at_1     0.5000
mean_top_score         0.8778
retrieve() p50         23.3 ms      p95 26.7 ms
```

`retrieval_mrr` and `retrieval_hit_at_1` are new in this module, added to
`prodllm.eval.runner`. They had to be: `retrieval_hit_rate` only asks whether
the gold article appeared *somewhere* in the top five, so it is blind to the
one thing a re-ranker is for. A second stage can leave the hit rate untouched
and still move the governing article from fifth place to first — and with the
generator reading passages in the order it is handed them, position is not
cosmetic.

Before building anything, the more important measurement: **how deep does the
first stage have to fetch before the gold article is present at all?** A
re-ranker can only promote a candidate it is shown, so this is a hard ceiling
on everything the second stage can achieve.

| candidate depth | recall | Arabic | English | questions still missing |
|---|---|---|---|---|
| 5 | 0.75 | 0.60 | 0.90 | q01, q09, q14, q17, q19 |
| 10 | 0.80 | 0.60 | **1.00** | q01, q09, q17, q19 |
| **20** | **0.90** | 0.80 | 1.00 | q01, q09 |
| 30 | 0.90 | 0.80 | 1.00 | q01, q09 |
| 50 | 0.95 | 0.90 | 1.00 | q01 |
| 100 | 0.95 | 0.90 | 1.00 | q01 |

This table is the reason `rerank_candidates` defaults to 20 and not to 5 or to
50. At 20 the ceiling is 0.90 against a current 0.75, which is real headroom;
going to 50 buys another 0.05 of ceiling at two and a half times the
cross-encoder cost per request. 20 is where the curve flattens before it steps
again.

Two findings fell out of it that are worth more than the number they produced.

**Retrieval failure here is almost entirely an Arabic problem.** English recall
saturates at 1.00 by depth 10. Arabic is at 0.60 at depth 10 and never exceeds
0.90. The corpus is indexed with one vector per language per chunk precisely so
that both languages hit, and English works; the Arabic side is where the
quality is being lost. That is consistent with the encoder being
`multilingual-e5-small` — 384 dimensions and a vocabulary shared across a
hundred languages — and with the Arabic text having been through PDF column
extraction and bidi repair before it was ever embedded.

**q01 is unreachable and no re-ranker will fix it.** q01 asks, in Arabic, for
the general prescription period; its gold article is 374. Its English twin q02
asks the same question and retrieves 374 at **rank 1**. In Arabic, 374 is not
in the top 100. The article is in the index and is findable — just not from
Arabic. This caps the honest ceiling at 0.95, and it localises the defect to
the Arabic embedding of either the query or that article, which is a Module 1
retrieval bug and not something a second stage can paper over.

### The teacher

`BAAI/bge-reranker-v2-m3`, **567 755 777 parameters**, an XLM-RoBERTa-large
cross-encoder with a single-logit relevance head. Multilingual was not a
preference: a question in Arabic has to be scored against a passage whose
Arabic and English halves are both present, and an English-only cross-encoder
would be a stronger model on paper and useless here.

Its cost is the problem. Measured on this CPU, at `max_seq_length=512`:

| torch threads | teacher throughput |
|---|---|
| 1 | 0.8 pairs/s |
| 16 | **3.1 pairs/s** |

The 1-thread figure is not a strawman — it is what the job did on first run,
because `TorchEmbedder.from_settings` applies `PRODLLM_TORCH_NUM_THREADS`,
which this project sets to **1** so that a served replica has predictable
latency and is scaled by adding replicas rather than threads. That is the right
policy for a request handler and exactly the wrong one for an offline batch
job, and it cost a 3.9× slowdown with fifteen cores sitting idle.
`scripts/rerank_teacher_scores.py` now lifts the cap explicitly, after the
embedder is constructed, with a comment saying why.

At 3.1 pairs/s, re-ranking 20 candidates inside a request would add roughly
**6.5 seconds** of CPU latency to a 23 ms retrieval. The teacher is not
deployable here at any setting. That is what distillation is for.

### The training signal

Distillation can only transfer an opinion the teacher actually holds, so the
scores it produced were checked before any student was trained.
`scripts/rerank_teacher_scores.py` built **600 pseudo-queries** — 293 English,
307 Arabic — by lifting 40–250 character clauses out of corpus articles, then
scored each against the **20** passages the vector index retrieves for it:
**12 000 teacher judgements** over 1030 distinct passages, in 123.4 minutes.
That is 1.6 pairs/s rather than the 3.1 above, because the quantization job was
holding half the cores at the time; the 3.1 figure is the uncontended one.

Two properties make those scores usable as labels:

| | vector ordering | teacher ordering |
|---|---|---|
| mean rank of the source article | 1.80 | **1.01** |
| MRR | 0.8778 | **0.9955** |
| hit@1 | 82.4 % | **99.1 %** |

*(556 of the 600 pseudo-queries retrieve their own source article inside the
top 20 at all; the other 44 are first-stage recall failures that no re-ranker
can repair, and they are excluded from this table rather than scored as zero.)*

First, the teacher disagrees with the vector index about the top result in
**138 of 600 lists (23 %)** and is right nearly every time it does — so there
is a real reordering to learn, not a permutation of equals. Second, the median
within-list spread of teacher logits is **15.86** (10th percentile 11.57), so
the softmax the student is fitting is sharply peaked rather than flat; a flat
target would train the student to output a constant.

What this table is **not** is a prediction of the gain on real questions. A
pseudo-query is a clause copied verbatim out of its own article, so its lexical
overlap with the correct passage is far higher than a user's question would
have, and 99.1 % hit@1 is a statement about an easy task. It establishes that
the labels carry signal. The honest measurement of what re-ranking buys is the
20-question evaluation below, on questions written independently of the corpus.

<!-- PENDING: distillation results, student vs teacher vs baseline table -->

## Quantization

The generative model is **`Qwen/Qwen2.5-1.5B-Instruct`**, 1 543 714 304
parameters, quantized to **4-bit GPTQ** with group size 128 by
`scripts/quantize_gptq.py`. It is a real checkpoint, not an estimate:
`artifacts/qwen2.5-1.5b-gptq-4bit`, **1096.6 MiB of weights against 2944.4 MiB
at BF16 — 2.69×** — produced in 87.4 minutes on CPU, with perplexity measured
on held-out corpus passages before and after. Every number in this half of the
report comes from `reports/eval/gptq_quantization.json`, which records the
hardware alongside the results.

### Why symmetric INT8 behaves badly on an LLM

Not asserted from the literature — measured on this model.
`scripts/quantization_analysis.py` hooks all **196 linear modules** across
Qwen2.5-1.5B's 28 decoder layers, pushes **16 corpus passages** through in
float32, accumulates per-input-channel absolute maxima, and then quantizes each
weight matrix three ways and reports the reconstruction error. Results in
`reports/eval/quant_analysis.json`.

**A symmetric per-tensor INT8 scale is set by the largest magnitude in the
tensor, and that largest magnitude is an outlier.** With one scale for
everything, 255 levels are spread across a range that one freak channel
defines; the values that carry most of the signal are then squeezed into a
handful of levels near zero. The measurement:

| | max ÷ median channel magnitude |
|---|---|
| **activations** (mean over 196 modules) | **73.3×** |
| activations, median module | 13.7× |
| activations, worst module (`layers.1.mlp.down_proj`) | **6994×** — absmax 5115.6 against a median of 0.73 |
| **weights** (mean over 196 modules) | **4.4×** |
| weights, worst module | 28.1× |

**That 73× against 4× is the whole story, and it decides the strategy.**
Activations are wildly outlier-ridden; weights are comparatively tame. So
quantizing *weights* is tractable and quantizing *activations* is what destroys
an LLM — which is why the deployed artifact here is **weight-only 4-bit** and
not INT8 W8A8. 166 of 196 modules have at least one activation channel above
ten times their median, so this is the normal case, not a tail.

**The epicentre is `mlp.down_proj`**, averaging **429.6×** against 4–17× for
the attention projections. That is the SwiGLU bottleneck: `down_proj` consumes
the elementwise product of the gate and up projections, and a product of two
activations has far heavier tails than either. Any scheme that quantizes
activations has to special-case this layer; several published INT8 methods do
exactly that, and it is a hint that the easier path is not to quantize
activations at all.

**Per-channel scaling fixes the weight half of the problem, and the fix is
unambiguous:** mean reconstruction error falls from **0.03929 to 0.00984**, a
**3.93×** improvement — and per-channel was better in **196 of 196 modules**,
not on average but in every single one. The worst case,
`layers.15.self_attn.q_proj`, improves **15.5×**. One scale per output channel
means a single bad channel can no longer tax the rest of the matrix.

**Attention is the part where error does not stay small.** `q_proj` and
`k_proj` feed a dot product that goes through a softmax, which exponentiates
its inputs: a small absolute error in a logit becomes a multiplicative error in
an attention weight, and the renormalisation redistributes that error across
every position in the context. A quantization error in an MLP is attenuated by
the next layer norm; a quantization error in attention changes *which tokens
the model reads*. Note that `v_proj` carries the **lowest** per-channel weight
error of any role (0.01020) and `k_proj` the lowest per-tensor error (0.02142)
— low weight error in attention is not the same as safety, which is why this
section ends at perplexity rather than at a table of reconstruction errors.

### Weight error is a weak proxy for model quality

Worth stating plainly, because the same JSON invites the wrong conclusion. Mean
weight reconstruction error, all three schemes:

| scheme | bits/weight | mean weight error |
|---|---|---|
| INT8 per-tensor symmetric | 8.0 | 0.03929 |
| INT8 per-channel symmetric | 8.00–8.02 | **0.00984** |
| INT4 group-wise, g=128 (round-to-nearest) | 4.25 | 0.10589 |

(Per-channel INT8 costs one fp32 scale per output channel, so its true rate
depends on the matrix: 8.021 bits/weight for the 1536-wide projections,
8.004 for the 8960-wide `down_proj`. Group-wise INT4 at g=128 costs a scale
*and* a zero point per 128 weights, hence 4.25 rather than 4.0.)

Read as a ranking of deployable options this says 4-bit is **10.8× worse** than
per-channel INT8 and should be avoided. The measured perplexity says otherwise:
4-bit cost **5.96 %**. Two reasons the table misleads. The INT4 row is plain
round-to-nearest, which is *not* what ships — GPTQ's error compensation
(above) optimises the layer's **output** on calibration inputs, and nobody
consumes the weights. And a model's quality is not linear in weight-space
distance at all: the 0.00984 scheme would still need its activations
quantized to be fast, and that is where the 73× lives.

This is the argument for weight-only 4-bit in one line: **quantize the tensor
whose outliers are 4×, leave the tensor whose outliers are 73× in floating
point, and compensate what is left in output space rather than weight space.**

### Why GPTQ here and not AWQ

The module permits either. AWQ was the first choice and was abandoned for
reasons that are worth recording, because they are about tooling reality rather
than about the algorithm:

* **AutoAWQ, the reference implementation, was archived in May 2025.** The vLLM
  documentation marks the library deprecated and points at `llm-compressor`
  instead. Its own deprecation note states the last tested configuration was
  Torch 2.6.0 with Transformers 4.51.3.
* **Neither AWQ path supports native Windows.** `autoawq-kernels` documents
  "Windows: must use WSL2"; the release wheels are Linux `x86_64`. The only
  WSL2 distribution on this machine is Docker Desktop's own, and the Docker
  daemon was not running.
* **`llm-compressor` resolves on Windows but demands `transformers>=5`**, and
  this project pins `transformers<5` because the serving path is tested against
  the 4.x API.

GPTQModel, by contrast, is maintained, documents Windows 11 and CPU support,
and ships a CPU kernel (`GPTQ_TORCH_CPU_AVX512` — and this Tiger Lake CPU has
AVX-512). So GPTQ is the 4-bit method that can be *executed and checked* on
this machine rather than described, which is the only reason it was preferred.

It still needs `transformers>=5`, so `scripts/quantize_gptq.py` runs in its own
interpreter: a PEP 723 inline dependency header plus `uv run --script` builds an
isolated, pinned environment for it. Nothing under `src/prodllm` imports
anything from it, and its only outputs are a checkpoint and a JSON record. The
hardware recorder is loaded from `src/prodllm/bench/hardware.py` *by file path*
rather than installed, which is why that module was written to depend on
nothing but the standard library, `psutil` and `nvidia-smi`.

### What GPTQ actually does

Round-to-nearest quantization treats each weight independently and accepts
whatever output error follows. GPTQ quantizes a layer column by column and,
after fixing each column, **updates the columns not yet quantized to
compensate for the error just introduced**, using the inverse Hessian of the
layer's inputs — estimated from calibration data — to decide how.

The consequence is the point: the layer's *output* on calibration-like inputs
is preserved much better than its *weights* are. Nobody consumes the weights.

Calibration therefore matters, and it is drawn from **this project's own
corpus** — Egyptian Civil Code articles, both languages. Calibrating on web
text and serving Arabic legal prose would estimate the Hessian from the wrong
input distribution. Calibration and perplexity-evaluation passages are
**disjoint and seeded**; measuring perplexity on the calibration set would
report how well the quantizer memorised its own calibration data.

### What the 4-bit run measured

| | float32 baseline | 4-bit GPTQ | change |
|---|---|---|---|
| perplexity (48 held-out passages, 10 545 tokens) | 15.3550 | **16.2697** | **+5.96 %** |
| weights on disk | 2944.4 MiB *(BF16 equivalent)* | **1096.6 MiB** | **÷2.69** |
| checkpoint directory | — | 1111.7 MiB | — |
| throughput, this CPU | 52.7 tok/s | 16.3 tok/s | **÷3.2** |

Settings: `bits=4`, `group_size=128`, `damp_percent=0.01`, `desc_act=false`,
96 calibration passages, 48 disjoint evaluation passages, `max_seq_length=512`,
seed 20261025. Hardware: Intel 16-core CPU, 31.7 GiB RAM, torch 2.14.1+cpu.

**The perplexity cost is the result worth having.** Dropping from 16 bits per
weight to 4.25 costs **6 % perplexity** on in-domain legal text — the model is
measurably worse and nowhere near broken, which is the trade the whole module
is about. Two caveats keep that honest: perplexity is not answer quality, which
is what the RAGAS section measures; and 48 passages is a small sample, so the
figure carries real uncertainty and is reported to two decimals because that is
what the harness printed, not because the third would mean anything.

**The throughput row is a measurement of this machine, not of 4-bit
inference,** and it points the wrong way on purpose. 4-bit is faster than FP16
on real deployments because a memory-bandwidth-bound decoder moves a quarter as
many bytes per token, and because fused CUDA kernels dequantize inside the
matmul. Neither applies here: there is no usable GPU, so the pure-torch backend
dequantizes each weight block to float and then multiplies, doing strictly more
arithmetic than the unquantized model. **Quantization on this box buys memory
and costs speed.** Reporting 16.3 tok/s as the cost of 4-bit, without that
sentence, would be the single most misleading number in this report.

### The VRAM table

Computed by `scripts/vram_table.py` from architecture dimensions, not quoted
from a model card. The parameter counts are derived and asserted in
`tests/test_bench.py`.

Note the 4-bit row costs **4.25 bits per weight**, not 4.00. Group-wise schemes
store a scale and a zero point per group of weights at higher precision; at the
standard `group_size=128` that is 32 extra bits per 128 weights = 0.25 bits per
weight. Ignoring it understates a 70B checkpoint by about two gigabytes.

**Qwen2.5-1.5B-Instruct** — 1 543 656 960 parameters, 28 layers, 2 KV heads,
tied embeddings. KV cache at 4 096 tokens, FP16: 0.11 GiB (28.0 KiB/token).

The derived count can be checked against the downloaded artifact, and it is
worth doing once to establish that the rest of the table is not fiction:
1 543 656 960 parameters × 2 bytes (the checkpoint is BF16) = **2 944.3 MiB**,
against an actual `model.safetensors` of **2 944.4 MiB**. The arithmetic and
the file agree to a tenth of a mebibyte.

| precision | bits/weight | weights | + KV | + runtime | total | fits 4 GiB? |
|---|---|---|---|---|---|---|
| FP32 | 32.00 | 5.75 GiB | 0.11 | 1.50 | 7.36 GiB | no |
| FP16 | 16.00 | 2.88 GiB | 0.11 | 1.50 | **4.48 GiB** | **no** |
| INT8 | 8.00 | 1.44 GiB | 0.11 | 1.50 | 3.05 GiB | yes |
| INT4 | 4.25 | 0.76 GiB | 0.11 | 1.50 | **2.37 GiB** | **yes** |

This is the clearest practical result in the module and it is about *this*
machine: a 1.5B model at FP16 **does not fit** a 4 GiB card once the CUDA
context, allocator and activations are counted, and at 4 bits it fits with
1.6 GiB to spare. Quantization is not a nicety on this hardware; it is the
difference between loading and not loading.

The `+ runtime` column is 1.5 GiB of CUDA context, allocator fragmentation,
cuBLAS workspaces and live activations. It is a working figure, stated
explicitly in `scripts/vram_table.py` so it can be argued with rather than
buried in a total.

### A 70B model on one 80 GB GPU

**Llama-3.1-70B** — 80 layers, hidden 8192, 64 attention heads, **8 KV heads**,
intermediate 28672, vocab 128 256, untied embeddings. Derived parameter count:
**70 553 706 496** (70.55 B). KV cache: 320.0 KiB per token.

| precision | weights | + KV (4k ctx) | + runtime | total | fits 80 GiB? |
|---|---|---|---|---|---|
| FP32 | 262.83 GiB | 1.25 | 1.50 | 265.58 | no |
| FP16 | 131.42 GiB | 1.25 | 1.50 | 134.17 | **no** |
| INT8 | 65.71 GiB | 1.25 | 1.50 | 68.46 | yes |
| INT4 | 34.91 GiB | 1.25 | 1.50 | 37.66 | yes |

And the question that actually matters for serving — what context length does
each precision leave room for?

| precision | weights | spare after runtime | FP16 KV cache tokens |
|---|---|---|---|
| FP16 | 131.4 GiB | −52.9 GiB | **will not load** |
| INT8 | 65.7 GiB | 12.8 GiB | 41 915 tokens |
| INT4 | 34.9 GiB | 43.6 GiB | **142 843 tokens** |

So: a 70B model **cannot** run on a single 80 GB GPU at FP16 — it is 51 GiB
over budget before a single token of cache. At INT8 it fits with about 42 k
tokens of context. At 4 bits it fits with about 143 k tokens, which is the
difference between a toy context and a useful one.

The deeper point is in the last column rather than the weights column.
**Quantizing weights to 4 bits does nothing to the KV cache** — it is left at
FP16 and grows linearly with context — so on a single-GPU deployment nearly all
of the memory that 4-bit frees gets re-spent on context length. A table of
weight sizes alone makes 4-bit look like pure headroom. It is a reallocation,
and grouped-query attention is doing as much work as the quantizer: 8 KV heads
against 64 query heads is an 8× cut in cache size, and without it the 4-bit row
would hold about 18 k tokens instead of 143 k.

## What is not measured here

Listed rather than buried, because on this hardware the gap between what the
module asks for and what can honestly be produced is wide enough that leaving
it implicit would be misleading.

**No 4-bit kernel speedup.** The speed benefit of 4-bit weights comes from CUDA
kernels that fuse dequantization into the matmul and cut memory traffic. There
is no reachable CUDA device here, so the quantized model is run through
GPTQModel's pure-torch backend, which dequantizes to float and multiplies. That
path is *slower* than float32, not faster. Every quality number from it is
real — the weights are the 4-bit weights — and the tokens/second figure is a
property of the fallback backend, not of 4-bit inference. It is reported with
that label attached and it is not the number a deployment would see.

**No tokens/second comparison across precisions on a GPU.** Same reason. The
VRAM table is arithmetic and is presented as arithmetic; the throughput column
the module asks for would have to be invented, so it is absent instead.

**The re-ranker's latency is a CPU latency.** A cross-encoder is embarrassingly
parallel over candidates and is normally served on a GPU in a single batched
forward pass. The figures here are 16-core CPU figures at batch size 8, which
is the deployment this project actually has.

**Pseudo-queries are not questions.** The student is distilled on clauses
lifted from the articles, because the twenty labelled questions are the test
set and there is no LLM available to generate synthetic ones. Training queries
are therefore declarative where test queries are interrogative. This is the
single biggest threat to the student's measured quality and the most likely
explanation if it underperforms its teacher by more than the capacity gap
predicts.

**The distillation stages are not in `dvc.yaml`.** The corpus and index stages
are; these are not. `dvc repro` runs every stage, and teacher scoring takes
about an hour, so adding them would turn a routine rebuild into an overnight
job. The two commands are documented in the README instead. This is a
deliberate trade and it does mean the re-ranker artifact has no recorded hash,
which is the same class of gap as the unlocked `build_index` stage already
noted in the README's limitations.

**Perplexity is measured on this corpus, not on a standard benchmark.** The
module asks for perplexity or task accuracy; perplexity here is over held-out
Egyptian Civil Code articles rather than WikiText. That makes it the right
number for *this* deployment and the wrong number to compare against a
published figure for the same model. Passages are scored individually rather
than concatenated, for reasons given in `scripts/quantize_gptq.py`.
