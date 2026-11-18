"""Sweep RAG chunking configurations and log every run to MLflow.

    uv run --group tracking --group eval python scripts/sweep_chunking.py

One MLflow run per configuration. Each run rebuilds the chunks from
``articles.json``, re-embeds and re-indexes them, answers the fixed
twenty-question set, judges the answers with RAGAS faithfulness, and logs the
parameters, the metrics and the recorded run itself as an artifact.

Why the whole index is rebuilt per configuration rather than the chunking being
varied at query time: chunk_size and overlap change what is *in* the index, so
a sweep that reused one index would be comparing six labels over identical
vectors. The rebuild is the expensive part and it is the part that matters.

The generator must be a real LLM. With the extractive generator every answer is
a verbatim copy of its contexts, so faithfulness is ~1.0 for every
configuration and the comparison is empty -- the script refuses to run that way
rather than producing a flat table that looks like a result.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlflow

from prodllm.config import Settings
from prodllm.corpus.chunk import ChunkConfig, chunk_corpus_with
from prodllm.corpus.schema import Article
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.eval.dataset import DEFAULT_QUESTION_SET, EvalQuestion, load_questions
from prodllm.eval.faithfulness import (
    DEFAULT_JUDGE_MODEL,
    FAITHFULNESS_THRESHOLD,
    build_judge,
    score_faithfulness,
)
from prodllm.eval.runner import run_evaluation
from prodllm.rag import RagEngine, build_generator

E5_SMALL = "intfloat/multilingual-e5-small"
E5_BASE = "intfloat/multilingual-e5-base"

DEFAULT_EXPERIMENT = "rag-chunking"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

# Six configurations over three axes, so that neither the embedding model nor
# the overlap is judged from a single run. `baseline` is exactly what Module 1
# shipped, which is what makes the rest of the table readable as a comparison
# rather than as a list.
SWEEP: tuple[ChunkConfig, ...] = (
    ChunkConfig(name="baseline", chunk_size=1500, overlap=0, embedding_model=E5_SMALL),
    ChunkConfig(name="narrow", chunk_size=600, overlap=0, embedding_model=E5_SMALL),
    ChunkConfig(
        name="narrow-overlap", chunk_size=600, overlap=120, embedding_model=E5_SMALL
    ),
    ChunkConfig(
        name="wide-overlap", chunk_size=1500, overlap=300, embedding_model=E5_SMALL
    ),
    ChunkConfig(name="base-model", chunk_size=1500, overlap=0, embedding_model=E5_BASE),
    ChunkConfig(
        name="base-model-overlap", chunk_size=600, overlap=120, embedding_model=E5_BASE
    ),
)


def settings_for(config: ChunkConfig, base: Settings) -> Settings:
    """Settings pinned to this configuration's embedding model.

    ``artifacts_dir`` is pointed at a per-model directory so that
    ``from_settings`` cannot find the baked-in e5-small weights and quietly
    serve them while the run claims to be using e5-base.
    """
    model_name = config.embedding_model.split("/")[-1]
    return base.model_copy(
        update={
            "source_model_id": config.embedding_model,
            "artifacts_dir": Path(".caches/sweep") / model_name,
            "allow_download": True,
        }
    )


def load_articles(path: Path) -> list[Article]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Article.model_validate(record) for record in records]


def run_one(
    config: ChunkConfig,
    articles: list[Article],
    questions: list[EvalQuestion],
    base: Settings,
    judge: Any,
    embedder_cache: dict[str, TorchEmbedder],
    artifact_dir: Path,
) -> dict[str, float]:
    print(f"\n=== {config.name} ===")
    print(
        f"  chunk_size={config.chunk_size} overlap={config.overlap} "
        f"model={config.embedding_model}"
    )

    chunks = chunk_corpus_with(articles, config)
    print(f"  chunks: {len(chunks)}")

    settings = settings_for(config, base)
    if config.embedding_model not in embedder_cache:
        print(f"  loading {config.embedding_model} ...")
        embedder_cache[config.embedding_model] = TorchEmbedder.from_settings(settings)
    embedder = embedder_cache[config.embedding_model]

    started = time.perf_counter()
    index = RagEngine.build_index([chunk.model_dump() for chunk in chunks], embedder)
    index_seconds = time.perf_counter() - started
    print(f"  indexed {len(index)} vectors in {index_seconds:.1f}s")

    engine = RagEngine(
        embedder=embedder,
        index=index,
        generator=build_generator(base),
        top_k=base.top_k,
        min_score=base.min_score,
    )

    print(f"  answering {len(questions)} questions ...")
    run = run_evaluation(
        engine, questions, params=config.as_params(), generator=base.llm_model
    )

    print("  judging faithfulness ...")
    faith = score_faithfulness(run, judge=judge)

    metrics: dict[str, float] = {
        **run.metrics,
        **faith.as_metrics(),
        "chunks": float(len(chunks)),
        "vectors": float(len(index)),
        "index_build_seconds": index_seconds,
        "embedding_dim": float(embedder.info.embedding_dim),
    }

    mlflow.log_params(
        {
            **config.as_params(),
            "judge_model": faith.judge_model,
            "generator_model": base.llm_model,
            "top_k": base.top_k,
        }
    )
    mlflow.log_metrics(metrics)

    run_path = run.save(artifact_dir / f"eval_run_{config.name}.json")
    mlflow.log_artifact(str(run_path), artifact_path="eval")
    mlflow.set_tag("config_name", config.name)
    mlflow.set_tag("gate", "pass" if faith.mean >= FAITHFULNESS_THRESHOLD else "fail")

    print(
        f"  faithfulness {faith.mean:.3f} over {faith.judged_count} judged"
        f"  | retrieval hit {metrics['retrieval_hit_rate']:.2f}"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--articles", type=Path, default=Path("data/processed/articles.json")
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTION_SET)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="The tracking server. Start it with scripts/mlflow_server.py.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these configuration names (default: all six).",
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("reports/eval"))
    args = parser.parse_args()

    if not args.articles.is_file():
        print(
            f"ERROR articles file not found: {args.articles}\n"
            "Run `dvc repro build_corpus`, or scripts/build_corpus.py directly.",
            file=sys.stderr,
        )
        return 2

    base = Settings()
    if not base.llm_configured:
        print(
            "ERROR no generator LLM configured.\n"
            "Faithfulness over extractive answers is ~1.0 by construction and "
            "would compare nothing. Set PRODLLM_LLM_BASE_URL, PRODLLM_LLM_MODEL "
            "and PRODLLM_LLM_API_KEY -- see .env.example.",
            file=sys.stderr,
        )
        return 2

    configs = list(SWEEP)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {config.name for config in SWEEP}
        if unknown:
            print(f"ERROR unknown configuration(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        configs = [config for config in configs if config.name in wanted]

    articles = load_articles(args.articles)
    questions = load_questions(args.questions)
    print(
        f"{len(articles)} articles, {len(questions)} questions, "
        f"{len(configs)} configurations"
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    judge = build_judge(model=args.judge_model)

    embedder_cache: dict[str, TorchEmbedder] = {}
    results: dict[str, dict[str, float]] = {}
    for config in configs:
        with mlflow.start_run(run_name=config.name):
            results[config.name] = run_one(
                config,
                articles,
                questions,
                base,
                judge,
                embedder_cache,
                args.artifact_dir,
            )

    print("\n=== comparison ===")
    header = f"{'config':<20} {'faith':>7} {'hit':>6} {'chunks':>7} {'dim':>5}"
    print(header)
    print("-" * len(header))
    for name, metrics in sorted(
        results.items(), key=lambda item: item[1]["faithfulness"], reverse=True
    ):
        print(
            f"{name:<20} {metrics['faithfulness']:>7.3f} "
            f"{metrics['retrieval_hit_rate']:>6.2f} "
            f"{metrics['chunks']:>7.0f} {metrics['embedding_dim']:>5.0f}"
        )
    print(f"\nTracking UI: {args.tracking_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
