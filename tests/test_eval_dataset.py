"""The fixed question set, and the guards that keep it fixed."""

import json

import pytest

from prodllm.eval.dataset import (
    DEFAULT_QUESTION_SET,
    EXPECTED_QUESTION_COUNT,
    load_questions,
)


def write_set(tmp_path, records):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def question(identifier="q1", **overrides):
    return {
        "id": identifier,
        "question": "ما هي مدة التقادم؟",
        "lang": "ar",
        "expected_articles": [374],
        **overrides,
    }


# ------------------------------------------------------- the committed set


def test_the_committed_question_set_is_the_size_the_gate_assumes():
    questions = load_questions(DEFAULT_QUESTION_SET)

    assert len(questions) == EXPECTED_QUESTION_COUNT


def test_the_committed_set_covers_both_languages():
    languages = {question.lang for question in load_questions(DEFAULT_QUESTION_SET)}

    assert languages == {"ar", "en"}


def test_every_committed_question_names_an_expected_article():
    # Without this the retrieval_hit_rate metric silently reads 1.0.
    for item in load_questions(DEFAULT_QUESTION_SET):
        assert item.expected_articles, f"{item.id} has no expected article"


# ------------------------------------------------------------- the guards


def test_a_shrunken_set_is_refused_rather_than_silently_gating_on_less(tmp_path):
    path = write_set(tmp_path, [question("q1"), question("q2")])

    with pytest.raises(ValueError, match="holds 2 questions, expected 20"):
        load_questions(path)


def test_the_size_check_can_be_waived_deliberately(tmp_path):
    path = write_set(tmp_path, [question("q1"), question("q2")])

    assert len(load_questions(path, expected_count=None)) == 2


def test_duplicate_ids_are_refused(tmp_path):
    path = write_set(tmp_path, [question("q1"), question("q1")])

    with pytest.raises(ValueError, match="duplicate question ids"):
        load_questions(path, expected_count=None)


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(question("q1")) + "\nnot json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"questions\.jsonl:2 is not valid JSON"):
        load_questions(path, expected_count=None)


def test_a_line_missing_a_required_field_names_its_line_number(tmp_path):
    path = write_set(tmp_path, [question("q1"), {"id": "q2"}])

    with pytest.raises(ValueError, match=r"questions\.jsonl:2 is missing"):
        load_questions(path, expected_count=None)


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"absent\.jsonl"):
        load_questions(tmp_path / "absent.jsonl")


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(question("q1")) + "\n\n" + json.dumps(question("q2")) + "\n",
        encoding="utf-8",
    )

    assert len(load_questions(path, expected_count=None)) == 2


def test_expected_articles_defaults_to_empty(tmp_path):
    path = write_set(tmp_path, [{"id": "q1", "question": "x"}])

    loaded = load_questions(path, expected_count=None)

    assert loaded[0].expected_articles == ()
    assert loaded[0].lang == "ar"
