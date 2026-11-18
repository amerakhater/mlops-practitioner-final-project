"""Measure retrieval quality and latency, with or without the second stage.

    # the single-stage baseline every earlier module was measured against
    uv run python scripts/eval_retrieval.py --label vector-only

    # the 568M teacher, and the distilled student that replaces it
    uv run python scripts/eval_retrieval.py --reranker BAAI/bge-reranker-v2-m3 \
        --label teacher
    uv run python scripts/eval_retrieval.py --reranker artifacts/reranker \
        --label student

    # how deep the first stage has to fetch for the gold article to be there
    uv run python scripts/eval_retrieval.py --recall-curve

Runs the fixed twenty-question set through real retrieval over the real index
and reports rank-aware metrics next to measured latency, because an optimization
module's deliverable is the trade-off and not either number alone.

**Why rank-aware metrics and not just the hit rate.** ``retrieval_hit_rate``
asks whether the gold article appeared anywhere in the top five. A re-ranker
can leave that completely unchanged while moving the gold article from fifth to
first, which is the whole point of having one -- the generator reads the
passages in the order it is given them, and the answer leads with whatever came
first. MRR and hit@1 are what make that visible.

**Why ``--recall-curve`` matters more than it looks.** A re-ranker can only
promote an article the first stage handed it, so recall at the candidate depth
is a hard ceiling on everything the second stage can achieve. Measuring it
first is what turns ``rerank_candidates`` from a guess into a decision.

This script does not judge faithfulness: that needs a paid LLM judge and lives
in scripts/eval_gate.py. Everything here runs offline on CPU.
"""

import argparse
import json
import sys
from pathlib import Path

from prodllm.bench import describe_hardware, measure
from prodllm.config import Settings, get_settings
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.eval.dataset import (
    DEFAULT_QUESTION_SET,
    EvalQuestion,
    load_questions,
)
from prodllm.eval.runner import run_evaluation
from prodllm.rag import RagEngine, VectorIndex, build_generator
from prodllm.rerank.cross_encoder import CrossEncoderReranker

DEFAULT_OUT_DIR = Path("reports/eval")
RECALL_DEPTHS = (5, 10, 20, 30, 50, 100)


def build_engine(
    settings: Settings,
    embedder: TorchEmbedder,
    index: VectorIndex,
    reranker_source: str | None,
    top_k: int,
    candidates: int,
    max_seq_length: int,
    batch_size: int,
) -> tuple[RagEngine, CrossEncoderReranker | None]:
    """A one- or two-stage engine, depending on ``reranker_source``."""
    reranker = None
    if reranker_source:
        reranker = CrossEncoderReranker.load(
            reranker_source,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            local_files_only=Path(reranker_source).is_dir(),
        )
    return RagEngine(
        embedder=embedder,
        index=index,
        # The configured generator, not a hardcoded extractive one. Retrieval
        # metrics do not depend on it, but RAGAS faithfulness does: an
        # extractive answer is a verbatim copy of its contexts and scores ~1.0
        # by construction, so a before/after faithfulness comparison recorded
        # that way would compare nothing. With no LLM configured this still
        # falls back to extractive, which is what makes the retrieval half of
        # this script runnable with no API key.
        generator=build_generator(settings),
        top_k=top_k,
        min_score=settings.min_score,
        reranker=reranker,
        rerank_candidates=candidates,
    ), reranker


def recall_curve(
    settings: Settings,
    embedder: TorchEmbedder,
    index: VectorIndex,
    questions: list[EvalQuestion],
) -> dict[str, dict]:
    """Hit rate as a function of candidate depth -- the re-ranker's ceiling."""
    rows: dict[str, dict] = {}
    for depth in RECALL_DEPTHS:
        engine = RagEngine(
            embedder=embedder, index=index, top_k=depth, min_score=settings.min_score
        )
        hits = {"ar": 0, "en": 0}
        totals = {"ar": 0, "en": 0}
        missing = []
        for question in questions:
            retrieved = {a.article_number for a in engine.retrieve(question.question)}
            found = bool(set(question.expected_articles) & retrieved)
            totals[question.lang] = totals.get(question.lang, 0) + 1
            hits[question.lang] = hits.get(question.lang, 0) + int(found)
            if not found:
                missing.append(question.id)
        total = sum(totals.values())
        rows[str(depth)] = {
            "recall": (sum(hits.values()) / total) if total else 0.0,
            "recall_ar": (hits["ar"] / totals["ar"]) if totals["ar"] else 0.0,
            "recall_en": (hits["en"] / totals["en"]) if totals["en"] else 0.0,
            "missing": missing,
        }
        print(
            f"  recall@{depth:<4d} overall {rows[str(depth)]['recall']:.2f}  "
            f"ar {rows[str(depth)]['recall_ar']:.2f}  "
            f"en {rows[str(depth)]['recall_en']:.2f}  missing {missing}",
            flush=True,
        )
    return rows


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTION_SET)
    parser.add_argument("--index", type=Path, default=settings.index_dir)
    parser.add_argument(
        "--reranker",
        default=None,
        help="Model id or directory. Omit for the single-stage baseline.",
    )
    parser.add_argument("--label", default=None, help="Names the output file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument(
        "--rerank-candidates", type=int, default=settings.rerank_candidates
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--recall-curve",
        action="store_true",
        help="Also measure recall at increasing depth (the second-stage ceiling).",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=5,
        help="Repeats for the per-question latency measurement.",
    )
    args = parser.parse_args()

    label = args.label or ("vector-only" if not args.reranker else "reranked")

    questions = load_questions(args.questions)
    hardware = describe_hardware()
    print(f"hardware : {hardware.summary()}")
    print(f"label    : {label}")

    embedder = TorchEmbedder.from_settings(settings)
    index = VectorIndex.load(args.index)

    curve = None
    if args.recall_curve:
        print("\nrecall by candidate depth (the re-ranker's ceiling):")
        curve = recall_curve(settings, embedder, index, questions)

    engine, reranker = build_engine(
        settings,
        embedder,
        index,
        args.reranker,
        args.top_k,
        args.rerank_candidates,
        args.max_seq_length,
        args.batch_size,
    )
    if reranker is not None:
        print(
            f"reranker : {reranker.info.model_id} "
            f"({reranker.info.parameters:,} params), "
            f"{args.rerank_candidates} candidates -> top {args.top_k}"
        )
    else:
        print(f"reranker : none, top {args.top_k}")

    params = {
        "stage": label,
        "reranker": args.reranker or "",
        "reranker_parameters": reranker.info.parameters if reranker else 0,
        "top_k": args.top_k,
        "rerank_candidates": args.rerank_candidates if reranker else 0,
        "max_seq_length": args.max_seq_length,
        "embedding_model": settings.source_model_id,
    }

    generator_name = settings.llm_model if settings.llm_configured else "extractive"
    print(f"generator: {generator_name}")

    print("\nrunning the question set...")
    run = run_evaluation(engine, questions, params=params, generator=generator_name)

    # The run above gives a mean latency over one pass. This gives percentiles
    # over repeats of one representative question, through the shared harness,
    # so the figure is comparable with every other table in the report.
    probe = questions[0].question
    stats, _ = measure(
        lambda: engine.retrieve(probe), warmup=2, repeats=args.latency_repeats
    )
    print(f"retrieve() on q01: {stats.summary()}")

    print("\nmetrics:")
    for key, value in run.metrics.items():
        print(f"  {key:22s} {value:.4f}")

    print("\nper question:")
    for record in run.records:
        print(
            f"  {record.id} {record.lang} exp={record.expected_articles} "
            f"got={record.retrieved_articles} rr={record.reciprocal_rank:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.out_dir / f"retrieval_{label}.json"
    run.save(run_path)

    summary = {
        "label": label,
        "params": params,
        "metrics": run.metrics,
        "retrieve_latency": stats.as_dict(),
        "recall_curve": curve,
        "hardware": hardware.as_dict(),
    }
    summary_path = args.out_dir / f"retrieval_{label}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nwrote {run_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
