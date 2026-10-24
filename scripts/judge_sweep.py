"""Re-judge every recorded run in the sweep, in one pass.

    uv run --group tracking --group eval python scripts/judge_sweep.py

The sweep judges each configuration as it finishes, which means six judging
sessions spread over an hour. The judge is not deterministic -- re-judging an
unchanged run has been measured moving its mean by about 0.013 -- so a
comparison assembled that way carries drift between sessions that is
indistinguishable from a difference between configurations.

This re-judges all of them against one judge instance, back to back, and
writes the result back to both the artifact and the MLflow run. It changes no
answers and re-embeds nothing: the recorded runs are the fixed input, and only
the scoring is redone. That makes the comparison a comparison of
configurations rather than of judging sessions.

It also fills in the per-question scores, so the artifact a reviewer opens
shows which questions carried the difference instead of only the mean.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from prodllm.eval.faithfulness import (
    DEFAULT_JUDGE_MODEL,
    FAITHFULNESS_THRESHOLD,
    build_judge,
    score_faithfulness,
)
from prodllm.eval.runner import EvalRun

DEFAULT_EXPERIMENT = "rag-chunking"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--artifact-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("reports/mlflow-comparison.md"),
        help="Where to write the generated comparison table.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"ERROR no experiment named {args.experiment!r}", file=sys.stderr)
        return 2

    runs = [
        run
        for run in client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time ASC"],
            max_results=100,
        )
        if "faithfulness" in run.data.metrics
    ]
    if not runs:
        print(f"ERROR no finished runs in {args.experiment!r}", file=sys.stderr)
        return 2

    judge = build_judge(model=args.judge_model)
    print(f"re-judging {len(runs)} run(s) with {args.judge_model}\n")

    header = f"{'config':<20} {'before':>8} {'after':>8} {'delta':>8}"
    print(header)
    print("-" * len(header))

    results: dict[str, float] = {}
    for run in runs:
        name = run.data.params.get("config_name", run.info.run_name)
        path = args.artifact_dir / f"eval_run_{name}.json"
        if not path.is_file():
            source = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=f"eval/eval_run_{name}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")

        recorded = EvalRun.load(path)
        before = run.data.metrics["faithfulness"]
        result = score_faithfulness(recorded, judge=judge)
        recorded.save(path)

        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_metrics(result.as_metrics())
            mlflow.log_artifact(str(path), artifact_path="eval")
            mlflow.set_tag(
                "gate", "pass" if result.mean >= FAITHFULNESS_THRESHOLD else "fail"
            )
            mlflow.set_tag("judging", "single-pass")

        results[name] = result.mean
        print(
            f"{name:<20} {before:>8.4f} {result.mean:>8.4f} "
            f"{result.mean - before:>+8.4f}"
        )

    table = write_comparison(client, experiment.experiment_id, args.comparison)
    print(f"\ncomparison written to {table}")

    best = max(results, key=results.get)
    print(f"best: {best} ({results[best]:.4f})")
    print("Promote it with: uv run --group tracking python scripts/register_best.py")
    return 0


def write_comparison(client: MlflowClient, experiment_id: str, path: Path) -> Path:
    """Render the sweep as a Markdown table, read straight from the runs.

    Generated rather than hand-written so the report cannot quietly disagree
    with the tracking server it claims to summarise.
    """
    runs = [
        run
        for run in client.search_runs(
            [experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.faithfulness DESC"],
            max_results=100,
        )
        if "faithfulness" in run.data.metrics
    ]

    columns = (
        "config",
        "chunk_size",
        "overlap",
        "embedding_model",
        "chunks",
        "vectors",
        "faithfulness",
        "retrieval hit",
        "index s",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for run in runs:
        params, metrics = run.data.params, run.data.metrics
        cells = (
            params.get("config_name", run.info.run_name),
            params.get("chunk_size", "?"),
            params.get("overlap", "?"),
            "`" + params.get("embedding_model", "?").split("/")[-1] + "`",
            f"{metrics.get('chunks', 0):.0f}",
            f"{metrics.get('vectors', 0):.0f}",
            f"**{metrics['faithfulness']:.3f}**",
            f"{metrics.get('retrieval_hit_rate', float('nan')):.2f}",
            f"{metrics.get('index_build_seconds', 0):.0f}",
        )
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!-- Generated by scripts/judge_sweep.py. Do not edit by hand. -->\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
