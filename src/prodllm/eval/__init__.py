"""Offline evaluation of the retrieval pipeline.

Kept out of the serving path on purpose. Nothing under ``prodllm.api`` or
``prodllm.rag`` imports this package, so the runtime image can omit ragas,
langchain and their dependency trees entirely -- and an evaluation harness
that cannot be imported at request time cannot slow a request down or take
the service out with it.
"""

from prodllm.eval.dataset import EvalQuestion, load_questions
from prodllm.eval.runner import EvalRecord, EvalRun, run_evaluation

__all__ = [
    "EvalQuestion",
    "EvalRecord",
    "EvalRun",
    "load_questions",
    "run_evaluation",
]
