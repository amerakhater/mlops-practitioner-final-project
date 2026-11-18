"""The fixed question set the faithfulness gate is defined on.

Fixed is the operative word. A gate measured against a question set that any
commit may quietly shrink is not a gate: drop the three questions that score
worst and the mean clears any threshold. ``load_questions`` therefore refuses
a set whose size has changed unless the caller states the new size.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUESTION_SET = Path("eval/questions.jsonl")

# The gate in the handbook is specified on twenty questions. Stated here so a
# question added or removed fails loudly rather than moving the threshold.
EXPECTED_QUESTION_COUNT = 20


@dataclass(frozen=True)
class EvalQuestion:
    """One question, with the articles a correct retrieval should surface."""

    id: str
    question: str
    lang: str
    expected_articles: tuple[int, ...] = ()


def load_questions(
    path: Path = DEFAULT_QUESTION_SET,
    expected_count: int | None = EXPECTED_QUESTION_COUNT,
) -> list[EvalQuestion]:
    """Read the JSONL question set.

    ``expected_count=None`` disables the size check, which is what the unit
    tests use; every production caller leaves it at the default.
    """
    if not path.is_file():
        raise FileNotFoundError(f"question set not found: {path}")

    questions: list[EvalQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
        try:
            questions.append(
                EvalQuestion(
                    id=record["id"],
                    question=record["question"],
                    lang=record.get("lang", "ar"),
                    expected_articles=tuple(record.get("expected_articles", ())),
                )
            )
        except KeyError as error:
            # A line that parses but omits a required field gets the same
            # treatment as one that does not parse. A bare KeyError here names
            # the field but not the line, which is the harder half to find.
            raise ValueError(
                f"{path}:{number} is missing the required field {error}"
            ) from error

    identifiers = [question.id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
        raise ValueError(f"{path} has duplicate question ids: {duplicates}")

    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(
            f"{path} holds {len(questions)} questions, expected {expected_count}. "
            "The faithfulness gate is defined on a fixed set -- update "
            "EXPECTED_QUESTION_COUNT deliberately if the set really did change."
        )
    return questions
