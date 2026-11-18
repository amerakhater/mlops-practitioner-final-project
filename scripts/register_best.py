"""Promote the best sweep run to Production in the MLflow Model Registry.

    uv run --group tracking python scripts/register_best.py
    uv run --group tracking python scripts/register_best.py --dry-run

Reads the sweep's experiment, picks the run with the highest RAGAS
faithfulness, logs that run's chunking configuration as a pyfunc model,
registers it, and moves the ``Production`` alias onto the new version.

Three things make this auditable rather than ceremonial:

* The model is logged **into the winning run**, not into a fresh one, so the
  registered version and the metrics that justified it share a run id.
* A run that fails the faithfulness gate is never promoted, even if it is the
  best of a bad set. "Best available" and "good enough to serve" are different
  questions and conflating them is how a regression ships.
* ``params.yaml`` is rewritten to match, so ``dvc repro`` rebuilds the corpus
  with the configuration that was actually promoted. Without that step the
  registry records one configuration and the pipeline builds another.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from prodllm.config import Settings
from prodllm.corpus.chunk import ChunkConfig
from prodllm.eval.faithfulness import FAITHFULNESS_THRESHOLD
from prodllm.tracking.chunking_model import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    log_chunking_model,
)

DEFAULT_EXPERIMENT = "rag-chunking"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
PARAMS_FILE = Path("params.yaml")

# The promoted run's recorded answers, committed so the CI faithfulness gate
# can judge them without the corpus, the index or a tracking server.
PROMOTED_RUN_FILE = Path("eval/promoted_run.json")

SELECTION_METRIC = "faithfulness"


def ranked_runs(client: MlflowClient, experiment_name: str) -> list:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(
            f"No experiment named {experiment_name!r}. Run the sweep first:\n"
            "  uv run --group tracking --group eval python scripts/sweep_chunking.py"
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
        max_results=50,
    )
    runs = [run for run in runs if SELECTION_METRIC in run.data.metrics]
    if not runs:
        raise SystemExit(
            f"No finished run in {experiment_name!r} has a {SELECTION_METRIC} metric."
        )
    return runs


def select_run(runs: list, config_name: str | None) -> tuple[object, object | None]:
    """The run to promote, and the top scorer when it is not the same one.

    Promoting the top of a ranking is only correct when the ranking means
    something. When the spread across configurations is smaller than the
    judge's own run-to-run noise -- which is the case for this sweep -- the
    ordering is not evidence, and picking on cost or on risk is the better
    decision. ``--config`` is how that decision gets made explicitly instead
    of by quietly redefining the selection metric until the intended run wins.
    """
    top = runs[0]
    if config_name is None:
        return top, None

    for run in runs:
        if run.data.params.get("config_name", run.info.run_name) == config_name:
            return run, (None if run.info.run_id == top.info.run_id else top)

    available = sorted(
        run.data.params.get("config_name", run.info.run_name) for run in runs
    )
    raise SystemExit(f"No finished run named {config_name!r}. Available: {available}")


def config_from_run(run: object) -> ChunkConfig:
    params = run.data.params
    missing = {"chunk_size", "overlap", "embedding_model"} - set(params)
    if missing:
        raise SystemExit(
            f"Run {run.info.run_id} is missing parameter(s) {sorted(missing)}; "
            "it was not produced by scripts/sweep_chunking.py."
        )
    return ChunkConfig(
        name=params.get("config_name", run.info.run_name or "unnamed"),
        chunk_size=int(params["chunk_size"]),
        overlap=int(params["overlap"]),
        embedding_model=params["embedding_model"],
    )


def write_params(config: ChunkConfig, path: Path = PARAMS_FILE) -> bool:
    """Point params.yaml at ``config``. Returns whether anything changed.

    The file is rewritten rather than regenerated so its explanatory comments
    survive -- a params file that loses its comments every promotion stops
    being read.
    """
    text = path.read_text(encoding="utf-8")
    current = yaml.safe_load(text)["chunking"]
    wanted = {
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "embedding_model": config.embedding_model,
    }
    if current == wanted:
        return False

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        for key, value in wanted.items():
            if stripped.startswith(f"{key}:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[index] = f"{indent}{key}: {value}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def export_promoted_run(
    run_id: str, config: ChunkConfig, destination: Path = PROMOTED_RUN_FILE
) -> Path:
    """Copy the winning run's recorded answers out of MLflow and commit them.

    Downloaded from the tracking server rather than copied from
    ``reports/eval/`` on disk: the local file may be from a later sweep than
    the run being promoted, and the whole point of the exported file is that
    it is the one the registered version was scored on.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"eval/eval_run_{config.name}.json"
    )
    destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def warn_on_encoder_mismatch(config: ChunkConfig) -> bool:
    """Shout if the promoted encoder is not the one the service defaults to.

    This is the quietest serious failure in the whole project. ``params.yaml``
    drives ``build_index``, but the running service picks its encoder from
    ``PRODLLM_SOURCE_MODEL_ID``. Promote a configuration built on e5-base
    while the service still defaults to e5-small and nothing breaks: the index
    loads, ``/health`` is green, every request returns five articles, and
    every one of them is the nearest neighbour in the wrong vector space.

    A warning rather than a refusal, because promoting a better encoder is a
    legitimate thing to want -- it just takes three coordinated changes.
    """
    served = Settings().source_model_id
    if config.embedding_model == served:
        return False

    print(
        f"\n!!! ENCODER MISMATCH\n"
        f"    promoted : {config.embedding_model}\n"
        f"    served   : {served}  (PRODLLM_SOURCE_MODEL_ID)\n"
        f"    The service would embed queries with one model and search an\n"
        f"    index built by another. Retrieval keeps working and silently\n"
        f"    returns nonsense. Before deploying this configuration, update\n"
        f"    the default in src/prodllm/config.py, the value in .env, and\n"
        f"    the weights baked in by scripts/fetch_weights.py -- all three.",
        file=sys.stderr,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default=PRODUCTION_ALIAS)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Promote this configuration by name instead of the top scorer. "
            "For when the spread between configurations is inside the judge's "
            "noise and the choice is therefore made on cost or risk."
        ),
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Recorded on the model version. Required with --config.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FAITHFULNESS_THRESHOLD,
        help="Refuse to promote a run scoring below this.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be promoted and change nothing.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    if args.config and not args.reason:
        print(
            "ERROR --config requires --reason. Overriding the ranking is a "
            "judgement call, and a registry that records the override without "
            "recording why is no better than one that records neither.",
            file=sys.stderr,
        )
        return 2

    runs = ranked_runs(client, args.experiment)
    run, outranked = select_run(runs, args.config)
    config = config_from_run(run)
    score = run.data.metrics[SELECTION_METRIC]

    if outranked is not None:
        top_score = outranked.data.metrics[SELECTION_METRIC]
        top_name = outranked.data.params.get("config_name", outranked.info.run_name)
        print(
            f"deliberate selection: promoting {config.name} ({score:.3f}) over "
            f"{top_name} ({top_score:.3f}), a gap of {top_score - score:.3f}.\n"
            f"  reason: {args.reason}\n"
        )

    print(f"promoting     : {config.name}  ({run.info.run_id})")
    print(f"  chunk_size  : {config.chunk_size}")
    print(f"  overlap     : {config.overlap}")
    print(f"  embedding   : {config.embedding_model}")
    print(f"  {SELECTION_METRIC:<12}: {score:.3f}")
    for name in ("retrieval_hit_rate", "grounded_rate", "chunks"):
        if name in run.data.metrics:
            print(f"  {name:<12}: {run.data.metrics[name]:.3f}")

    if score < args.threshold:
        print(
            f"\nREFUSING to promote: {score:.3f} is below the gate "
            f"({args.threshold}). The best run of a bad sweep is still a bad "
            "configuration -- fix retrieval before promoting.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing registered, params.yaml untouched.")
        return 0

    # Resuming the winning run is what ties the registered version to the
    # metrics that justified it.
    with mlflow.start_run(run_id=run.info.run_id):
        info = log_chunking_model(config, registered_model_name=args.model_name)

    version = client.search_model_versions(
        f"name='{args.model_name}'", order_by=["version_number DESC"], max_results=1
    )[0]
    client.set_registered_model_alias(args.model_name, args.alias, version.version)
    client.set_model_version_tag(
        args.model_name, version.version, "faithfulness", f"{score:.4f}"
    )
    client.set_model_version_tag(
        args.model_name, version.version, "config_name", config.name
    )
    if args.reason:
        client.set_model_version_tag(
            args.model_name, version.version, "selection_reason", args.reason
        )
    if outranked is not None:
        client.set_model_version_tag(
            args.model_name,
            version.version,
            "outranked",
            outranked.data.params.get("config_name", outranked.info.run_name),
        )
    client.update_model_version(
        args.model_name,
        version.version,
        description=(
            f"Chunking configuration {config.name}: chunk_size={config.chunk_size}, "
            f"overlap={config.overlap}, embedding_model={config.embedding_model}. "
            f"RAGAS faithfulness {score:.3f} on the fixed 20-question set."
            + (f"\n\nSelected deliberately: {args.reason}" if args.reason else "")
        ),
    )

    print(f"\nregistered    : {args.model_name} v{version.version}")
    print(f"alias         : {args.alias} -> v{version.version}")
    print(f"model uri     : models:/{args.model_name}@{args.alias}")
    print(f"logged at     : {info.model_uri}")

    exported = export_promoted_run(run.info.run_id, config)
    print(f"gate input    : {exported}  (commit this)")

    warn_on_encoder_mismatch(config)

    if write_params(config):
        print(f"{PARAMS_FILE} updated -- run `dvc repro` to rebuild with it.")
    else:
        print(f"{PARAMS_FILE} already matches the promoted configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
