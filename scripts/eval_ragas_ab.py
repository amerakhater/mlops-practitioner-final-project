"""RAGAS faithfulness, before and after the re-ranker, judged by one judge.

    # 1. record both runs with a real generator configured
    export PRODLLM_LLM_BASE_URL=https://openrouter.ai/api/v1
    export PRODLLM_LLM_MODEL=openai/gpt-4o-mini
    export PRODLLM_LLM_API_KEY=...
    uv run python scripts/eval_retrieval.py --label before
    uv run python scripts/eval_retrieval.py --label after \
        --reranker artifacts/reranker

    # 2. judge them together
    uv run --group eval python scripts/eval_ragas_ab.py \
        --before reports/eval/retrieval_before.json \
        --after  reports/eval/retrieval_after.json

Judges two recorded runs with the **same judge model in one invocation**. That
is the whole reason this exists rather than running scripts/eval_gate.py twice:
reports/module-2.md records that this judge is not deterministic, so two
separate judging sessions differ by more than a small re-ranking effect would.
Scoring both runs in one process at least holds the judge fixed.

**Both runs must come from a real LLM generator.** The extractive generator
copies its contexts verbatim, so its faithfulness is ~1.0 by construction and a
before/after comparison of two such runs compares nothing.
``score_faithfulness`` refuses those runs outright, and this script surfaces
that refusal rather than passing ``allow_extractive``.

**A correct refusal scores zero, and that is reported separately.** Module 2
found this and it matters more here than it did there: when retrieval misses,
a well-behaved generator says so, and RAGAS scores an answer with no support in
its contexts at 0.0 -- which is the right answer to the question RAGAS asks and
the wrong summary of the system's behaviour. Since the re-ranker changes *which*
questions retrieval gets right, the refusal count changes too, and a raw mean
faithfulness delta would silently mix "the generator became more faithful" with
"retrieval missed a different number of times". So the mean is reported three
ways: over everything judged, over questions where retrieval hit, and over
questions where it missed. The middle one is the generation-quality comparison.
"""

import argparse
import json
import sys
from pathlib import Path

from prodllm.eval.faithfulness import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    JudgeUnavailableError,
    build_judge,
    score_faithfulness,
)
from prodllm.eval.runner import EvalRun

DEFAULT_OUT = Path("reports/eval/ragas_ab.json")


def summarise(run: EvalRun, scores: dict[str, float]) -> dict:
    """Faithfulness split by whether retrieval actually found the article."""
    judged = [r for r in run.records if r.id in scores]
    hits = [r for r in judged if r.retrieval_hit]
    misses = [r for r in judged if not r.retrieval_hit]

    def mean(records: list) -> float | None:
        if not records:
            return None
        return sum(scores[r.id] for r in records) / len(records)

    return {
        "generator": run.generator,
        "questions": len(run.records),
        "judged": len(judged),
        "faithfulness_all": mean(judged),
        "faithfulness_on_retrieval_hits": mean(hits),
        "faithfulness_on_retrieval_misses": mean(misses),
        "retrieval_hits": len(hits),
        "retrieval_misses": len(misses),
        "zero_scored": sum(1 for r in judged if scores[r.id] == 0.0),
        "retrieval_metrics": run.metrics,
        "per_question": {r.id: scores[r.id] for r in judged},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    args = parser.parse_args()

    for path in (args.before, args.after):
        if not path.is_file():
            print(f"ERROR run not found: {path}", file=sys.stderr)
            return 1

    before = EvalRun.load(args.before)
    after = EvalRun.load(args.after)

    for label, run in (("before", before), ("after", after)):
        if run.generator in ("", "extractive"):
            print(
                f"ERROR the '{label}' run was recorded with generator "
                f"{run.generator!r}. RAGAS faithfulness over extractive answers "
                "is ~1.0 by construction and compares nothing. Configure "
                "PRODLLM_LLM_BASE_URL, PRODLLM_LLM_MODEL and "
                "PRODLLM_LLM_API_KEY, re-record both runs with "
                "scripts/eval_retrieval.py, then judge them.",
                file=sys.stderr,
            )
            return 2

    # One judge object, shared, so the two runs are scored by the same thing.
    try:
        judge = build_judge(args.judge_model, args.judge_base_url)
    except JudgeUnavailableError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 3

    print(f"judge  : {args.judge_model} via {args.judge_base_url}")
    print(f"before : {args.before}  (generator {before.generator})")
    print(f"after  : {args.after}  (generator {after.generator})")

    print("\njudging the 'before' run...")
    before_result = score_faithfulness(before, judge=judge, model=args.judge_model)
    print("judging the 'after' run...")
    after_result = score_faithfulness(after, judge=judge, model=args.judge_model)

    before_summary = summarise(before, before_result.per_question)
    after_summary = summarise(after, after_result.per_question)

    def delta(key: str) -> float | None:
        a, b = before_summary[key], after_summary[key]
        return None if a is None or b is None else b - a

    payload = {
        "schema_version": 1,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "before": before_summary,
        "after": after_summary,
        "delta": {
            "faithfulness_all": delta("faithfulness_all"),
            "faithfulness_on_retrieval_hits": delta("faithfulness_on_retrieval_hits"),
            "retrieval_mrr": (
                after_summary["retrieval_metrics"].get("retrieval_mrr", 0)
                - before_summary["retrieval_metrics"].get("retrieval_mrr", 0)
            ),
        },
        "unjudged": {
            "before": before_result.unjudged,
            "after": after_result.unjudged,
        },
    }

    print(f"\n{'metric':<36s} {'before':>10s} {'after':>10s} {'delta':>10s}")
    print("-" * 70)
    for key in (
        "faithfulness_all",
        "faithfulness_on_retrieval_hits",
        "faithfulness_on_retrieval_misses",
    ):
        a, b = before_summary[key], after_summary[key]
        shown = f"{b - a:+.4f}" if a is not None and b is not None else "n/a"
        print(
            f"{key:<36s} "
            f"{'n/a' if a is None else f'{a:.4f}':>10s} "
            f"{'n/a' if b is None else f'{b:.4f}':>10s} "
            f"{shown:>10s}"
        )
    for key in ("retrieval_hits", "retrieval_misses", "zero_scored"):
        a, b = before_summary[key], after_summary[key]
        print(f"{key:<36s} {a:>10d} {b:>10d} {b - a:>+10d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # The per-question scores were written back onto the runs by
    # score_faithfulness, so re-saving preserves the breakdown and not only
    # the mean -- a reviewer can then see whether one bad question is carrying
    # the whole difference.
    before.save(args.before)
    after.save(args.after)
    print(f"\nwrote {args.out}")
    print(f"updated {args.before} and {args.after} with per-question scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
