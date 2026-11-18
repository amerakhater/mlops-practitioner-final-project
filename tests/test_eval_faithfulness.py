"""The faithfulness result, the key resolution, and the extractive refusal.

No test here calls a judge. The scoring path is exercised with a stub metric,
so the suite stays runnable with no API key, no network and without the eval
dependency group installed -- the same property the rest of the suite has with
no weights and no index.
"""

import sys
import types

import pytest

from prodllm.eval import faithfulness as module
from prodllm.eval.faithfulness import (
    FAITHFULNESS_THRESHOLD,
    FaithfulnessResult,
    JudgeUnavailableError,
    resolve_api_key,
    score_faithfulness,
)
from prodllm.eval.runner import EvalRecord, EvalRun

JUDGE_KEYS = (
    "PRODLLM_JUDGE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "PRODLLM_LLM_API_KEY",
)


@pytest.fixture
def no_keys(monkeypatch):
    """A process with no judge credentials anywhere, including in .env."""
    for name in JUDGE_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "Settings", lambda: _SettingsStub(""))
    return monkeypatch


class _SettingsStub:
    def __init__(self, key):
        self.llm_api_key = key


def record(identifier="q1", contexts=("[Article 374] ...",), **overrides):
    defaults = {
        "id": identifier,
        "question": "q?",
        "lang": "en",
        "answer": "a",
        "contexts": list(contexts),
        "retrieved_articles": [374],
        "expected_articles": [374],
        "top_score": 0.9,
        "grounded": bool(contexts),
        "latency_ms": 1.0,
    }
    return EvalRecord(**{**defaults, **overrides})


# ----------------------------------------------------------------- result


def test_the_mean_is_over_the_judged_questions():
    result = FaithfulnessResult(
        per_question={"a": 1.0, "b": 0.5}, judge_model="m", unjudged=["c"]
    )

    assert result.mean == 0.75
    assert result.judged_count == 2


def test_an_empty_result_scores_zero_rather_than_dividing_by_nothing():
    assert FaithfulnessResult(per_question={}, judge_model="m").mean == 0.0


def test_the_gate_reads_the_threshold():
    result = FaithfulnessResult(per_question={"a": 0.8}, judge_model="m")

    assert result.passes(threshold=0.75)
    assert not result.passes(threshold=0.9)


def test_the_gate_fails_a_run_that_dodged_questions():
    # 1.0 on the two it answered, nothing on the other eighteen.
    result = FaithfulnessResult(
        per_question={"a": 1.0, "b": 1.0}, judge_model="m", unjudged=["c"] * 18
    )

    assert result.passes(threshold=0.75)
    assert not result.passes(threshold=0.75, required_count=20)


def test_metrics_report_both_halves_of_the_picture():
    result = FaithfulnessResult(
        per_question={"a": 1.0, "b": 0.0}, judge_model="m", unjudged=["c"]
    )

    assert result.as_metrics() == {
        "faithfulness": 0.5,
        "faithfulness_judged": 2.0,
        "faithfulness_unjudged": 1.0,
    }


def test_the_documented_threshold_is_the_handbook_number():
    assert FAITHFULNESS_THRESHOLD == 0.75


# ------------------------------------------------------------ key lookup


def test_an_explicit_key_wins(no_keys):
    assert resolve_api_key("explicit") == "explicit"


def test_the_judge_specific_variable_is_preferred(no_keys):
    no_keys.setenv("OPENROUTER_API_KEY", "router")
    no_keys.setenv("PRODLLM_JUDGE_API_KEY", "judge")

    assert resolve_api_key() == "judge"


def test_the_generator_key_is_the_last_resort(no_keys):
    no_keys.setattr(module, "Settings", lambda: _SettingsStub("from-dotenv"))

    assert resolve_api_key() == "from-dotenv"


def test_a_missing_key_is_an_error_not_a_default(no_keys):
    with pytest.raises(JudgeUnavailableError, match="never be committed"):
        resolve_api_key()


def test_surrounding_whitespace_is_stripped(no_keys):
    no_keys.setenv("OPENROUTER_API_KEY", "  spaced  ")

    assert resolve_api_key() == "spaced"


# ---------------------------------------------------------------- scoring


class StubMetric:
    """Stands in for ragas Faithfulness, returning one score per input."""

    def __init__(self, scores):
        self._scores = scores
        self.seen: list[dict] = []

    def batch_score(self, inputs):
        self.seen = inputs
        return [type("R", (), {"value": score})() for score in self._scores]


@pytest.fixture
def stub_ragas(monkeypatch):
    """Stand in for ragas' Faithfulness metric.

    A synthetic module in sys.modules rather than a patch of the installed
    one. ``score_faithfulness`` imports Faithfulness at call time, and the
    import machinery returns an already-present sys.modules entry without
    importing its parent packages -- so this works whether or not the eval
    group is installed. That matters: the group is isolated on purpose
    (pyproject.toml), so the test job installs only `dev` and there is no real
    ragas to patch. These tests exercise score_faithfulness' own logic against
    StubMetric, which never needed the real metric anyway.
    """

    def install(scores):
        metric = StubMetric(scores)
        collections = types.ModuleType("ragas.metrics.collections")
        collections.Faithfulness = lambda llm: metric
        monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)
        return metric

    return install


def test_an_extractive_run_is_refused_rather_than_scored_trivially():
    run = EvalRun(records=[record()], generator="extractive")

    with pytest.raises(ValueError, match="verbatim copies of their contexts"):
        score_faithfulness(run, judge=object())


def test_an_extractive_run_can_be_scored_when_that_is_asked_for(stub_ragas):
    stub_ragas([1.0])
    run = EvalRun(records=[record()], generator="extractive")

    result = score_faithfulness(run, judge=object(), allow_extractive=True)

    assert result.mean == 1.0


def test_records_with_no_context_are_listed_rather_than_scored(stub_ragas):
    metric = stub_ragas([1.0, 0.0])
    run = EvalRun(
        records=[
            record("answered"),
            record("empty", contexts=()),
            record("also-answered"),
        ],
        generator="openai/gpt-4o-mini",
    )

    result = score_faithfulness(run, judge=object())

    assert result.unjudged == ["empty"]
    assert set(result.per_question) == {"answered", "also-answered"}
    assert len(metric.seen) == 2


def test_the_judge_receives_question_answer_and_contexts(stub_ragas):
    metric = stub_ragas([1.0])
    run = EvalRun(records=[record()], generator="openai/gpt-4o-mini")

    score_faithfulness(run, judge=object())

    assert metric.seen[0] == {
        "user_input": "q?",
        "response": "a",
        "retrieved_contexts": ["[Article 374] ..."],
    }


def test_the_scores_are_written_back_onto_the_run(stub_ragas):
    # Saving the run afterwards is what makes the per-question breakdown part
    # of the committed artifact rather than console output nobody keeps.
    stub_ragas([1.0, 0.5])
    run = EvalRun(records=[record("a"), record("b")], generator="openai/gpt-4o-mini")

    score_faithfulness(run, judge=object())

    assert [item.faithfulness for item in run.records] == [1.0, 0.5]


def test_an_unjudged_record_keeps_a_null_score(stub_ragas):
    stub_ragas([1.0])
    run = EvalRun(
        records=[record("a"), record("empty", contexts=())],
        generator="openai/gpt-4o-mini",
    )

    score_faithfulness(run, judge=object())

    assert run.records[0].faithfulness == 1.0
    assert run.records[1].faithfulness is None


def test_a_run_with_nothing_judgeable_needs_no_judge(stub_ragas):
    stub_ragas([])
    run = EvalRun(records=[record("empty", contexts=())], generator="m")

    result = score_faithfulness(run, judge=object())

    assert result.per_question == {}
    assert result.mean == 0.0
