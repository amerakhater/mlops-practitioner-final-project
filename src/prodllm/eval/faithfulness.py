"""RAGAS faithfulness over a recorded run.

Faithfulness asks one question: is every claim in the answer supported by the
contexts the generator was actually given? It needs no ground-truth answer,
which is why it is the metric this corpus can be gated on -- writing reference
answers for 1149 articles of statute is not a thing a side project does well,
and a gate built on bad references is worse than no gate.

One consequence is worth stating plainly: **the extractive generator scores
~1.0 by construction**, because its answer is a verbatim copy of its contexts.
A sweep run extractively would compare five configurations that all score a
perfect 1.0 and learn nothing. The evaluation therefore requires a real
generator LLM, and ``score_faithfulness`` refuses a run that was recorded
extractively rather than reporting a meaningless number.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from prodllm.config import Settings
from prodllm.eval.runner import EvalRun

# The gate in the handbook. Below this on the fixed question set, CI fails.
FAITHFULNESS_THRESHOLD = 0.75

# A cheap judge with reliable structured output -- ragas drives the metric
# through instructor, so a model that cannot return a schema reliably produces
# retries and NaNs rather than a low score.
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"

EXTRACTIVE_GENERATOR = "extractive"


class JudgeUnavailableError(RuntimeError):
    """No API key, so faithfulness cannot be measured.

    Raised rather than defaulted to 0.0 or 1.0: a missing key is a broken
    harness, and a gate that silently passes or fails on a configuration
    mistake teaches people to ignore it.
    """


@dataclass
class FaithfulnessResult:
    """Per-question faithfulness plus the mean the gate is read from."""

    per_question: dict[str, float]
    judge_model: str
    unjudged: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(self.per_question.values()) / len(self.per_question)

    @property
    def judged_count(self) -> int:
        return len(self.per_question)

    def passes(
        self,
        threshold: float = FAITHFULNESS_THRESHOLD,
        required_count: int | None = None,
    ) -> bool:
        """Whether this result clears the gate.

        ``required_count`` guards the obvious way around a mean: a
        configuration that retrieves nothing for half the set would otherwise
        be judged only on the half it answered, and could clear the threshold
        while being markedly worse.
        """
        if required_count is not None and self.judged_count < required_count:
            return False
        return self.mean >= threshold

    def as_metrics(self) -> dict[str, float]:
        return {
            "faithfulness": self.mean,
            "faithfulness_judged": float(self.judged_count),
            "faithfulness_unjudged": float(len(self.unjudged)),
        }


def resolve_api_key(explicit: str | None = None) -> str:
    """The judge's API key, from the argument, the environment or settings.

    ``PRODLLM_JUDGE_API_KEY`` first, so the judge can be pointed at a different
    account than the generator -- judging answers with the same key that wrote
    them is fine here, but it should not be the only option. ``OPENROUTER_API_KEY``
    and ``OPENAI_API_KEY`` follow, then the generator's own key, which is what
    picks the value up out of ``.env`` rather than requiring it to be exported
    into the environment by hand.

    The key is never read from a committed file: ``.env`` is gitignored and
    ``.env.example`` carries no value.
    """
    key = (
        explicit
        or os.environ.get("PRODLLM_JUDGE_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or Settings().llm_api_key
        or ""
    ).strip()
    if not key:
        raise JudgeUnavailableError(
            "No judge API key. Set OPENROUTER_API_KEY (or PRODLLM_JUDGE_API_KEY) "
            "in the environment, or PRODLLM_LLM_API_KEY in .env -- it is a "
            "secret and must never be committed."
        )
    return key


def build_judge(
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    api_key: str | None = None,
) -> Any:
    """An instructor-backed ragas LLM pointed at an OpenAI-compatible endpoint.

    Asynchronous on purpose. ``Faithfulness.batch_score`` fans the twenty
    questions out concurrently and refuses a synchronous client outright
    ("Cannot use agenerate() with a synchronous client"); it also turns a
    sweep's judging phase from twenty serial round trips into one.
    """
    import openai
    from ragas.llms import llm_factory

    client = openai.AsyncOpenAI(api_key=resolve_api_key(api_key), base_url=base_url)
    return llm_factory(model, provider="openai", client=client)


def score_faithfulness(
    run: EvalRun,
    judge: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    api_key: str | None = None,
    allow_extractive: bool = False,
) -> FaithfulnessResult:
    """Judge every answered question in ``run``.

    Records with no contexts are not judged: faithfulness to an empty context
    set is undefined, and scoring them either way would be an invention. They
    are listed in ``unjudged`` so the caller can see how many there were.
    """
    if run.generator == EXTRACTIVE_GENERATOR and not allow_extractive:
        raise ValueError(
            "This run was recorded with the extractive generator, whose answers "
            "are verbatim copies of their contexts -- faithfulness would be ~1.0 "
            "by construction and would compare nothing. Configure "
            "PRODLLM_LLM_BASE_URL and PRODLLM_LLM_MODEL and record the run "
            "again, or pass allow_extractive=True if a trivial score is what "
            "you actually want."
        )

    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=judge or build_judge(model, base_url, api_key))

    judgeable = [record for record in run.records if record.contexts]
    unjudged = [record.id for record in run.records if not record.contexts]

    scores: dict[str, float] = {}
    if judgeable:
        results = metric.batch_score(
            [
                {
                    "user_input": record.question,
                    "response": record.answer,
                    "retrieved_contexts": record.contexts,
                }
                for record in judgeable
            ]
        )
        for record, result in zip(judgeable, results, strict=True):
            scores[record.id] = float(result.value)

    # Write the scores back onto the run, so saving it afterwards preserves
    # the breakdown rather than only the mean.
    for record in run.records:
        record.faithfulness = scores.get(record.id)

    return FaithfulnessResult(
        per_question=scores,
        judge_model=model,
        unjudged=unjudged,
    )
