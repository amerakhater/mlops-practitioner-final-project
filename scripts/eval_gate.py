"""The RAGAS faithfulness quality gate. Exit code 1 means the build fails.

    uv run --group eval python scripts/eval_gate.py
    uv run --group eval python scripts/eval_gate.py --threshold 0.8

Judges a recorded evaluation run -- the one produced by the promoted
configuration -- against the fixed twenty-question set, and fails below the
threshold.

**What this gate does and does not cover.** It re-judges answers that were
recorded when the configuration was promoted; it does not re-run retrieval.
That is a consequence of the DVC remote being a local directory: CI cannot
`dvc pull` the corpus, so it cannot rebuild the index, so it cannot generate
fresh answers. What the gate therefore catches is a promoted configuration
whose answers are not supported by their own retrieved contexts, plus any
tampering with the committed run or the question set. What it cannot catch is
a retrieval regression introduced after promotion. Pointing the DVC remote at
cloud storage is what lifts that limitation -- see reports/module-2.md.

The judging is non-deterministic, so the gate is a threshold and not an
equality check, and a borderline configuration will flake. Promote with
headroom rather than tuning the threshold down to whatever today's run scored.
"""

import argparse
import sys
from pathlib import Path

from prodllm.eval.dataset import DEFAULT_QUESTION_SET, load_questions
from prodllm.eval.faithfulness import (
    DEFAULT_JUDGE_MODEL,
    FAITHFULNESS_THRESHOLD,
    JudgeUnavailableError,
    score_faithfulness,
)
from prodllm.eval.runner import EvalRun

DEFAULT_RUN = Path("eval/promoted_run.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTION_SET)
    parser.add_argument("--threshold", type=float, default=FAITHFULNESS_THRESHOLD)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--allow-unjudged",
        type=int,
        default=0,
        help=(
            "How many questions may go unjudged for want of retrieved context. "
            "Zero by default: a configuration that retrieves nothing for a "
            "question would otherwise raise the mean by not being scored on it."
        ),
    )
    args = parser.parse_args()

    if not args.run.is_file():
        print(
            f"ERROR recorded run not found: {args.run}\n"
            "It is written by scripts/register_best.py when a configuration is "
            "promoted, and committed so this gate can run without the corpus.",
            file=sys.stderr,
        )
        return 2

    questions = load_questions(args.questions)
    run = EvalRun.load(args.run)

    # The run and the question set must describe the same twenty questions.
    # Without this, editing questions.jsonl while leaving the recorded run
    # alone would leave the gate silently measuring the old set.
    recorded = {record.id for record in run.records}
    expected = {question.id for question in questions}
    if recorded != expected:
        print(
            f"ERROR {args.run} does not match {args.questions}.\n"
            f"  only in the run      : {sorted(recorded - expected)}\n"
            f"  only in the question set: {sorted(expected - recorded)}\n"
            "Re-record the run against the current question set.",
            file=sys.stderr,
        )
        return 2

    print(f"run        : {args.run}  ({len(run)} questions)")
    print(f"recorded   : {run.created_at}")
    print(f"generator  : {run.generator}")
    print(f"config     : {run.params}")
    print(f"judge      : {args.judge_model}")

    try:
        result = score_faithfulness(run, model=args.judge_model)
    except JudgeUnavailableError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2

    print(f"\nfaithfulness : {result.mean:.3f}  (threshold {args.threshold})")
    print(f"judged       : {result.judged_count}/{len(run)}")
    if result.unjudged:
        print(f"unjudged     : {result.unjudged}")

    worst = sorted(result.per_question.items(), key=lambda item: item[1])[:5]
    print("\nlowest-scoring questions:")
    for question_id, score in worst:
        print(f"  {question_id}  {score:.3f}")

    if len(result.unjudged) > args.allow_unjudged:
        print(
            f"\nFAIL {len(result.unjudged)} question(s) retrieved no context, "
            f"above the allowance of {args.allow_unjudged}.",
            file=sys.stderr,
        )
        return 1

    if result.mean < args.threshold:
        print(
            f"\nFAIL faithfulness {result.mean:.3f} is below {args.threshold}.",
            file=sys.stderr,
        )
        return 1

    print(f"\nPASS faithfulness {result.mean:.3f} >= {args.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
