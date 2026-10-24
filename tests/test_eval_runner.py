"""The recorded evaluation run: its metrics, and its on-disk form.

The serialised run is what CI judges, so the round trip and the schema guard
matter as much as the metrics do -- a gate reading a file it misunderstands is
worse than a gate that is absent.
"""

import json

import pytest

from prodllm.eval.dataset import EvalQuestion
from prodllm.eval.runner import SCHEMA_VERSION, EvalRecord, EvalRun, run_evaluation


def record(identifier="q1", **overrides):
    defaults = {
        "id": identifier,
        "question": "ما هي مدة التقادم؟",
        "lang": "ar",
        "answer": "خمس عشرة سنة.",
        "contexts": ["[Article 374] ..."],
        "retrieved_articles": [374, 375],
        "expected_articles": [374],
        "top_score": 0.88,
        "grounded": True,
        "latency_ms": 42.0,
    }
    return EvalRecord(**{**defaults, **overrides})


# ----------------------------------------------------------- retrieval hit


def test_a_hit_needs_only_one_of_the_expected_articles():
    assert record(retrieved_articles=[999, 374]).retrieval_hit


def test_a_miss_is_a_miss():
    assert not record(retrieved_articles=[1, 2, 3]).retrieval_hit


def test_a_question_with_no_expected_articles_counts_as_a_hit():
    # It still contributes to faithfulness; it must not drag retrieval down
    # for a target that was never stated.
    assert record(expected_articles=[]).retrieval_hit


# ---------------------------------------------------------------- metrics


def test_metrics_average_over_every_record():
    run = EvalRun(
        records=[
            record("q1", retrieved_articles=[374], latency_ms=100.0, top_score=0.9),
            record("q2", retrieved_articles=[1], latency_ms=200.0, top_score=0.7),
        ]
    )

    metrics = run.metrics

    assert metrics["questions"] == 2
    assert metrics["retrieval_hit_rate"] == 0.5
    assert metrics["mean_latency_ms"] == 150.0
    assert metrics["mean_top_score"] == pytest.approx(0.8)
    assert metrics["grounded_rate"] == 1.0


def test_metrics_of_an_empty_run_are_empty_rather_than_a_division_error():
    assert EvalRun(records=[]).metrics == {}


def test_faithfulness_is_not_among_the_free_metrics():
    # It costs money and a network round trip; it is attached by the judge.
    assert "faithfulness" not in EvalRun(records=[record()]).metrics


def test_created_at_is_stamped_automatically():
    assert EvalRun(records=[record()]).created_at


# ------------------------------------------------------------- round trip


def test_a_run_survives_being_written_and_read(tmp_path):
    original = EvalRun(
        records=[record("q1"), record("q2", grounded=False, contexts=[])],
        params={"chunk_size": 1500, "overlap": 0},
        generator="openai/gpt-4o-mini",
    )

    path = original.save(tmp_path / "nested" / "run.json")
    restored = EvalRun.load(path)

    assert restored.to_dict() == original.to_dict()
    assert restored.params == {"chunk_size": 1500, "overlap": 0}
    assert restored.generator == "openai/gpt-4o-mini"
    assert len(restored) == 2


def test_the_saved_file_keeps_arabic_readable(tmp_path):
    path = EvalRun(records=[record()]).save(tmp_path / "run.json")

    assert "التقادم" in path.read_text(encoding="utf-8")


def test_a_run_written_by_a_future_schema_is_refused(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "records": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema version"):
        EvalRun.load(path)


# --------------------------------------------------------------- the pass


class StubAnswer:
    def __init__(self, answer, sources, grounded=True):
        self.answer = answer
        self.sources = sources
        self.grounded = grounded


class StubSource:
    def __init__(self, number, score):
        self.article_number = number
        self.citation = f"Egyptian Civil Code, Article {number}"
        self.text_ar = "نص عربي"
        self.text_en = "English text"
        self.score = score


class StubEngine:
    def __init__(self, answer):
        self._answer = answer
        self.asked: list[str] = []

    def answer(self, question):
        self.asked.append(question)
        return self._answer


def test_run_evaluation_records_one_entry_per_question():
    engine = StubEngine(StubAnswer("ok", [StubSource(374, 0.9), StubSource(375, 0.8)]))
    questions = [
        EvalQuestion(id="q1", question="first?", lang="en", expected_articles=(374,)),
        EvalQuestion(id="q2", question="second?", lang="ar", expected_articles=(1,)),
    ]

    run = run_evaluation(engine, questions, params={"a": 1}, generator="model-x")

    assert engine.asked == ["first?", "second?"]
    assert [record.id for record in run.records] == ["q1", "q2"]
    assert run.params == {"a": 1}
    assert run.generator == "model-x"
    assert run.records[0].retrieved_articles == [374, 375]
    assert run.records[0].top_score == 0.9
    assert run.records[0].latency_ms >= 0


def test_a_context_carries_the_citation_and_both_languages():
    engine = StubEngine(StubAnswer("ok", [StubSource(374, 0.9)]))

    run = run_evaluation(
        engine, [EvalQuestion(id="q1", question="q?", lang="en")], generator="m"
    )

    context = run.records[0].contexts[0]
    assert "Egyptian Civil Code, Article 374" in context
    assert "نص عربي" in context
    assert "English text" in context


def test_an_ungrounded_answer_records_no_contexts_and_a_zero_top_score():
    engine = StubEngine(StubAnswer("nothing found", [], grounded=False))

    run = run_evaluation(
        engine, [EvalQuestion(id="q1", question="q?", lang="en")], generator="m"
    )

    assert run.records[0].contexts == []
    assert run.records[0].top_score == 0.0
    assert run.records[0].grounded is False
