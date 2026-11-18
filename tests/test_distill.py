"""The distillation scripts' logic: pseudo-query extraction, the loss, pruning.

These live in ``scripts/`` rather than ``src/`` because they are offline jobs
that nothing imports, but the pieces tested here are not glue. The loss decides
what the student learns, the layer pruning decides what the student *is*, and
the clause extractor decides what it is trained on. A silent fault in any of
the three produces a student that trains to a plausible-looking loss curve and
re-ranks badly, which the retrieval metrics would show only after an hour of
teacher scoring.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    """Import a file from scripts/, which is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1 = load_script("rerank_teacher_scores")
stage2 = load_script("distill_reranker")


# ------------------------------------------------------ pseudo-query extraction


def test_clauses_keeps_only_spans_in_the_query_length_window():
    short = "Too short."
    good = "A" * 80
    long_clause = "B" * 400

    found = stage1.clauses(f"{short} {good}. {long_clause}.")

    assert good in found
    assert short.rstrip(".") not in found
    assert long_clause not in found


def test_clauses_splits_on_arabic_punctuation():
    """The corpus separates sub-clauses with the Arabic comma, not the Latin one."""
    first = "ا" * 60
    second = "ب" * 60

    found = stage1.clauses(f"{first}،{second}")

    assert first in found
    assert second in found


def test_clauses_strips_the_sub_clause_numbering_the_parser_leaves_in():
    """Western and Arabic-Indic digits both appear in this corpus."""
    body = "x" * 60

    assert stage1.clauses(f"(1) {body}.") == [body]
    assert stage1.clauses(f"(١) {body}.") == [body]


def test_clauses_of_empty_text_is_empty():
    assert stage1.clauses("") == []
    assert stage1.clauses(None) == []


def test_clauses_collapses_runs_of_whitespace_from_pdf_extraction():
    """Column-aware PDF extraction leaves multiple spaces between words."""
    body = " ".join(["word"] * 20)

    found = stage1.clauses("word    " + "   ".join(["word"] * 19) + ".")

    assert found == [body]


def test_clauses_treats_a_newline_as_a_clause_boundary():
    """Deliberate, and safe: no chunk in this corpus contains a newline.

    Verified against data/processed/chunks.json -- 0 of 1092 chunks have a
    newline in either language, because the parser joins the extracted lines
    of an article into one string. So the newline in the split pattern only
    ever separates genuinely distinct blocks.
    """
    first, second = "a" * 60, "b" * 60

    assert stage1.clauses(f"{first}\n{second}") == [first, second]


def test_passage_key_is_stable_and_distinguishes_passages():
    assert stage1.passage_key("abc") == stage1.passage_key("abc")
    assert stage1.passage_key("abc") != stage1.passage_key("abd")


# -------------------------------------------------------------- the listwise loss


def test_listwise_kl_is_zero_when_the_student_matches_the_teacher():
    logits = torch.tensor([2.0, 0.5, -1.0])

    assert float(stage2.listwise_kl(logits, logits)) == pytest.approx(0.0, abs=1e-6)


def test_listwise_kl_ignores_a_constant_offset():
    """The reason for KL over a softmax instead of MSE on raw scores.

    A student that reproduces every teacher score shifted by a constant has
    ranked the list perfectly. MSE would punish it; this must not.
    """
    teacher = torch.tensor([2.0, 0.5, -1.0])
    student = teacher + 7.0

    assert float(stage2.listwise_kl(student, teacher)) == pytest.approx(0.0, abs=1e-6)


def test_listwise_kl_punishes_a_reversed_ordering_more_than_a_flat_one():
    teacher = torch.tensor([3.0, 0.0, -3.0])

    reversed_loss = float(stage2.listwise_kl(torch.tensor([-3.0, 0.0, 3.0]), teacher))
    flat_loss = float(stage2.listwise_kl(torch.tensor([0.0, 0.0, 0.0]), teacher))

    assert reversed_loss > flat_loss > 0


def test_listwise_kl_is_larger_for_a_confident_teacher_than_a_flat_one():
    """A list the teacher has no opinion about should contribute little."""
    wrong = torch.tensor([0.0, 0.0, 3.0])

    confident = float(stage2.listwise_kl(wrong, torch.tensor([6.0, 0.0, -6.0])))
    uncertain = float(stage2.listwise_kl(wrong, torch.tensor([0.1, 0.0, -0.1])))

    assert confident > uncertain


def test_listwise_kl_is_finite_for_widely_separated_logits():
    """log_softmax must not underflow to -inf and poison the gradient."""
    loss = stage2.listwise_kl(torch.tensor([-60.0, 60.0]), torch.tensor([60.0, -60.0]))

    assert math.isfinite(float(loss))


# ----------------------------------------------------------------- layer pruning


class _Layer(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))


class _Encoder(torch.nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(_Layer(i) for i in range(count))


class _Config:
    def __init__(self, layers: int) -> None:
        self.num_hidden_layers = layers


class _Model(torch.nn.Module):
    """Mimics the base_model.encoder.layer path of a HF encoder."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.encoder = _Encoder(count)
        self.config = _Config(count)

    @property
    def base_model(self) -> "_Model":
        return self


def test_truncate_keeps_a_contiguous_stack_from_the_embeddings_up():
    model = _Model(12)

    kept = stage2.truncate_layers(model, 4)

    assert kept == 4
    assert [layer.index for layer in model.encoder.layer] == [0, 1, 2, 3]


def test_truncate_updates_the_config_so_save_pretrained_round_trips():
    """Without this the saved config claims 12 layers and reload mismatches."""
    model = _Model(12)

    stage2.truncate_layers(model, 4)

    assert model.config.num_hidden_layers == 4


def test_truncate_to_more_layers_than_exist_is_a_no_op():
    model = _Model(6)

    kept = stage2.truncate_layers(model, 99)

    assert kept == 6
    assert len(model.encoder.layer) == 6
    assert model.config.num_hidden_layers == 6


def test_truncate_actually_removes_the_parameters():
    model = _Model(12)
    before = sum(p.numel() for p in model.parameters())

    stage2.truncate_layers(model, 3)

    assert sum(p.numel() for p in model.parameters()) < before
