"""Run the RAG pipeline over the fixed question set and record what it did.

The recorded run is the unit everything downstream consumes: MLflow logs its
metrics, RAGAS judges its answers against its contexts, and CI re-judges a
committed copy. Keeping it a plain serialisable record rather than a live
pipeline is what lets the gate run in CI, where neither the index nor the
corpus is available.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prodllm.eval.dataset import EvalQuestion
from prodllm.rag import RagEngine

SCHEMA_VERSION = 1


@dataclass
class EvalRecord:
    """What the pipeline did with one question."""

    id: str
    question: str
    lang: str
    answer: str
    contexts: list[str]
    retrieved_articles: list[int]
    expected_articles: list[int]
    top_score: float
    grounded: bool
    latency_ms: float
    # Attached after the fact by the judge, so the recorded run carries the
    # per-question scores and not only the mean. Without it the comparison is
    # auditable only down to one number per configuration, and a reviewer
    # cannot see that one bad question is carrying the whole difference.
    faithfulness: float | None = None

    @property
    def retrieval_hit(self) -> bool:
        """True when at least one expected article was retrieved.

        Any-of rather than all-of: several articles can answer one question,
        and the question set records the clearest one, not an exhaustive list.
        A question with no expected articles counts as a hit so that it
        contributes to faithfulness without distorting the retrieval number.
        """
        if not self.expected_articles:
            return True
        return bool(set(self.expected_articles) & set(self.retrieved_articles))

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first expected article, or 0.0 if none was retrieved.

        Rank-aware, and that is the point of recording it. ``retrieval_hit``
        asks only whether the right article appeared somewhere in the top-k,
        so it is blind to the one thing a re-ranker is for: moving the right
        article from position four to position one. A second stage can leave
        the hit rate completely unchanged and still be the difference between
        an answer that leads with the governing article and one that buries
        it under three near-misses -- and with a generator reading the
        passages in order, position is not cosmetic.
        """
        expected = set(self.expected_articles)
        if not expected:
            return 1.0
        for position, number in enumerate(self.retrieved_articles, start=1):
            if number in expected:
                return 1.0 / position
        return 0.0

    @property
    def hit_at_1(self) -> bool:
        """True when the top-ranked article is an expected one."""
        if not self.expected_articles:
            return True
        if not self.retrieved_articles:
            return False
        return self.retrieved_articles[0] in set(self.expected_articles)


@dataclass
class EvalRun:
    """A full pass over the question set, plus the configuration that produced it."""

    records: list[EvalRecord]
    params: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    generator: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def metrics(self) -> dict[str, float]:
        """Everything measurable without an LLM judge.

        Faithfulness is deliberately absent: it costs money and needs a
        network, so it is attached by the judge rather than computed here.
        """
        if not self.records:
            return {}
        total = len(self.records)
        return {
            "retrieval_hit_rate": sum(r.retrieval_hit for r in self.records) / total,
            # Rank-aware, so a re-ranker that reorders the same five articles
            # is visible here even when the hit rate does not move at all.
            "retrieval_mrr": sum(r.reciprocal_rank for r in self.records) / total,
            "retrieval_hit_at_1": sum(r.hit_at_1 for r in self.records) / total,
            "grounded_rate": sum(r.grounded for r in self.records) / total,
            "mean_top_score": sum(r.top_score for r in self.records) / total,
            "mean_latency_ms": sum(r.latency_ms for r in self.records) / total,
            "mean_contexts": sum(len(r.contexts) for r in self.records) / total,
            "questions": float(total),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "generator": self.generator,
            "params": self.params,
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalRun":
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"eval run schema version {version!r} cannot be read by this "
                f"build, which writes version {SCHEMA_VERSION}"
            )
        return cls(
            records=[EvalRecord(**record) for record in payload["records"]],
            params=payload.get("params", {}),
            schema_version=version,
            created_at=payload.get("created_at", ""),
            generator=payload.get("generator", ""),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalRun":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _context_text(article: Any) -> str:
    """The passage text handed to the judge.

    Both languages, because the generator saw both and faithfulness asks
    whether the answer is supported by what the generator was given. Handing
    the judge only the English half would mark an Arabic answer unfaithful for
    a context the model did in fact receive.
    """
    parts = [f"[{article.citation}]"]
    if article.text_ar:
        parts.append(article.text_ar)
    if article.text_en:
        parts.append(article.text_en)
    return "\n".join(parts)


def run_evaluation(
    engine: RagEngine,
    questions: list[EvalQuestion],
    params: dict[str, Any] | None = None,
    generator: str = "",
) -> EvalRun:
    """Ask every question and record the answer, the contexts and the timing."""
    records: list[EvalRecord] = []
    for question in questions:
        started = time.perf_counter()
        answer = engine.answer(question.question)
        latency_ms = (time.perf_counter() - started) * 1000
        records.append(
            EvalRecord(
                id=question.id,
                question=question.question,
                lang=question.lang,
                answer=answer.answer,
                contexts=[_context_text(source) for source in answer.sources],
                retrieved_articles=[s.article_number for s in answer.sources],
                expected_articles=list(question.expected_articles),
                top_score=max((s.score for s in answer.sources), default=0.0),
                grounded=answer.grounded,
                latency_ms=latency_ms,
            )
        )
    return EvalRun(records=records, params=dict(params or {}), generator=generator)
