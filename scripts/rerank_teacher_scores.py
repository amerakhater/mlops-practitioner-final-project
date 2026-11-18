"""Stage 1 of re-ranker distillation: record what the teacher thinks.

    uv run python scripts/rerank_teacher_scores.py --max-queries 400
    uv run python scripts/rerank_teacher_scores.py --limit 8 --out /tmp/smoke.json

Writes a training file holding, for each pseudo-query, the candidate articles
the real first stage retrieves for it and the score the 568M teacher gives each
one. Stage 2 (``scripts/distill_reranker.py``) reads that file and never loads
the teacher, which is the whole reason the two are separate: teacher inference
is the expensive half and it only has to happen once per corpus.

**Where the training queries come from.** The labelled question set is twenty
questions and it is the *test* set -- training on it would make every number in
reports/module-4.md a measurement of memorisation. There is no LLM available to
generate synthetic questions either (see reports/module-4.md), so the queries
here are clauses lifted out of the articles themselves, in the manner of the
Inverse Cloze Task: a span of in-domain legal text stands in for a question
about that text.

What that buys and what it costs, stated plainly:

* It needs **no labels at all**. The teacher's score is the target, so the
  pseudo-query only has to be in-domain and plausible -- it does not have to
  have a known correct answer. Distillation learns the teacher's *function*,
  not relevance from scratch.
* The candidate lists come from the **real FAISS index**, so the student is
  trained on exactly the distribution of hard negatives it will be asked to
  sort at serving time. This matters more than query realism: a re-ranker's job
  is to order the first stage's mistakes.
* The cost is a **register mismatch**. A clause from Article 374 is declarative
  and the eval questions are interrogative, so the training queries are not
  distributed like the test queries. This is the main threat to the student's
  quality and is called out as such in the report rather than hidden.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

from prodllm.bench.hardware import describe_hardware
from prodllm.config import get_settings
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.rag import RagEngine, VectorIndex, rerank_payload
from prodllm.rerank.cross_encoder import DEFAULT_TEACHER_MODEL, CrossEncoderReranker

DEFAULT_OUT = Path("data/processed/rerank_teacher_scores.json")

# Clause boundaries in both scripts: Latin and Arabic full stops, question
# marks and semicolons, plus the Arabic comma, which in this corpus separates
# the numbered sub-clauses of an article rather than items in a list.
_CLAUSE_SPLIT = re.compile(r"[.!?؟؛،\n]+")

# A pseudo-query has to be long enough to carry a legal proposition and short
# enough to look like a query. Below 40 characters the clauses are fragments
# like "وفي هذه الحالة"; above 250 they are the whole article, which would make
# the task trivially lexical.
MIN_QUERY_CHARS = 40
MAX_QUERY_CHARS = 250


def clauses(text: str) -> list[str]:
    """Candidate pseudo-queries from one article's text in one language."""
    found = []
    for raw in _CLAUSE_SPLIT.split(text or ""):
        clause = " ".join(raw.split())
        # Strip the sub-clause numbering the parser leaves in -- "(١)" and
        # "(2)" are not part of the proposition and appear in every article,
        # so leaving them in would put the same token in most queries.
        clause = re.sub(r"^[(\[]?[0-9٠-٩]+[)\]]?\s*", "", clause)
        if MIN_QUERY_CHARS <= len(clause) <= MAX_QUERY_CHARS:
            found.append(clause)
    return found


def passage_key(payload: str) -> str:
    """A stable id for a passage, so the file stores each one once."""
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=settings.corpus_chunks_path)
    parser.add_argument("--index", type=Path, default=settings.index_dir)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--candidates",
        type=int,
        default=settings.rerank_candidates,
        help="Candidates retrieved and scored per pseudo-query.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Teacher input length. The dominant cost here: attention is "
            "quadratic in it, so halving this roughly quarters the run."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "torch intra-op threads for this offline job (0 = all cores). "
            "Overrides the serving cap of PRODLLM_TORCH_NUM_THREADS."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Cap the number of pseudo-queries (0 = one per chunk per language).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Use only the first N chunks (smoke test)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261025,
        help="Seeds the clause choice, so the training set is reproducible.",
    )
    args = parser.parse_args()

    if not args.chunks.is_file():
        print(f"ERROR corpus chunks not found: {args.chunks}", file=sys.stderr)
        return 1

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if args.limit:
        chunks = chunks[: args.limit]

    rng = random.Random(args.seed)
    queries: list[tuple[str, str, int]] = []  # (query, lang, source_article)
    for chunk in chunks:
        article = int(chunk.get("article_number", 0))
        for lang, field in (("ar", "text_ar"), ("en", "text_en")):
            options = clauses(chunk.get(field, ""))
            if options:
                queries.append((rng.choice(options), lang, article))

    # Shuffle before truncating, so a --max-queries cap samples the whole code
    # rather than the first few hundred articles. The corpus is ordered by
    # article number and the books are thematic, so taking a prefix would train
    # the student on obligations and never show it property or succession.
    rng.shuffle(queries)
    if args.max_queries:
        queries = queries[: args.max_queries]

    if not queries:
        print("ERROR no pseudo-queries could be built", file=sys.stderr)
        return 1

    hardware = describe_hardware()
    print(f"hardware   : {hardware.summary()}")
    print(f"chunks     : {len(chunks)}")
    print(f"queries    : {len(queries)}")
    print(f"candidates : {args.candidates} per query")
    print(f"pairs      : ~{len(queries) * args.candidates}")
    print(f"teacher    : {args.teacher} @ max_seq_length={args.max_seq_length}")

    embedder = TorchEmbedder.from_settings(settings)
    index = VectorIndex.load(args.index)

    # Undo the serving thread cap, deliberately and after the embedder is
    # loaded. ``TorchEmbedder.from_settings`` applies
    # PRODLLM_TORCH_NUM_THREADS, which this project sets to 1 so that a served
    # replica has predictable latency and is scaled by adding replicas. That is
    # the right policy for a request handler and exactly the wrong one here:
    # this is a one-off offline batch job that owns the machine, and leaving it
    # at 1 thread measured 0.8 teacher pairs/s against 16 cores sitting idle --
    # a ~3 hour run instead of a ~20 minute one.
    torch.set_num_threads(args.threads or (os.cpu_count() or 1))
    print(f"torch threads : {torch.get_num_threads()}")
    # top_k is the candidate depth here: this engine is a retrieval harness,
    # not a serving path, so it returns the whole pool for the teacher to sort.
    engine = RagEngine(
        embedder=embedder, index=index, top_k=args.candidates, min_score=0.0
    )
    teacher = CrossEncoderReranker.load(
        args.teacher,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    print(f"teacher params: {teacher.info.parameters:,}")

    passages: dict[str, str] = {}
    examples: list[dict] = []
    started = time.perf_counter()
    scored_pairs = 0

    for number, (query, lang, source_article) in enumerate(queries, start=1):
        candidates = engine.retrieve(query)
        if not candidates:
            continue
        payloads = [rerank_payload(article) for article in candidates]
        scores = teacher.score(query, payloads)
        scored_pairs += len(payloads)

        keys = []
        for payload in payloads:
            key = passage_key(payload)
            passages[key] = payload
            keys.append(key)

        examples.append(
            {
                "query": query,
                "lang": lang,
                "source_article": source_article,
                "candidates": keys,
                "candidate_articles": [a.article_number for a in candidates],
                "vector_scores": [round(a.score, 6) for a in candidates],
                "teacher_scores": [round(float(s), 6) for s in scores],
            }
        )

        if number % 10 == 0 or number == len(queries):
            elapsed = time.perf_counter() - started
            rate = scored_pairs / elapsed if elapsed else 0.0
            remaining = (len(queries) - number) * args.candidates
            eta_min = (remaining / rate / 60) if rate else 0.0
            print(
                f"  {number}/{len(queries)} queries, {scored_pairs} pairs, "
                f"{rate:.1f} pairs/s, eta {eta_min:.0f} min",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "teacher": args.teacher,
                "teacher_parameters": teacher.info.parameters,
                "teacher_max_seq_length": args.max_seq_length,
                "candidates": args.candidates,
                "seed": args.seed,
                "scored_pairs": scored_pairs,
                "elapsed_seconds": round(elapsed, 1),
                "hardware": hardware.as_dict(),
                "passages": passages,
                "examples": examples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {args.out}")
    print(f"  examples     : {len(examples)}")
    print(f"  passages     : {len(passages)} distinct")
    print(f"  scored pairs : {scored_pairs}")
    print(
        f"  elapsed      : {elapsed / 60:.1f} min "
        f"({scored_pairs / elapsed:.1f} pairs/s)"
        if elapsed
        else ""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
