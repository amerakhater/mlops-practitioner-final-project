"""The serving embedder, tested with a mocked model and tokenizer.

All weight loading lives in ``from_settings``, so ``__init__`` takes ready-made
objects and the tokenize -> forward -> pool -> normalize path is fully testable
with no weights on disk. This is the module most worth covering, and these
mocks are what make it reachable in a fresh clone.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from prodllm.config import Settings
from prodllm.embedding import torch_embedder as module_under_test
from prodllm.embedding.base import ModelInfo
from prodllm.embedding.torch_embedder import TorchEmbedder

DIM = 8


def make_tokenizer() -> MagicMock:
    """A tokenizer that pads to the batch longest and truncates at max_length.

    One token per whitespace-separated word, which is enough to assert padding,
    truncation and prefix handling without a 16 MB vocabulary.
    """

    def tokenize(texts, padding=True, truncation=True, max_length=512, **_):
        lengths = [max(1, min(len(t.split()), max_length)) for t in texts]
        seq = max(lengths)
        input_ids = torch.zeros((len(texts), seq), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), seq), dtype=torch.long)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return MagicMock(side_effect=tokenize)


def make_model(dim: int = DIM) -> MagicMock:
    """A model returning a deterministic last_hidden_state of the right shape."""

    def forward(input_ids=None, attention_mask=None, **_):
        batch, seq = input_ids.shape
        generator = torch.Generator().manual_seed(int(input_ids.sum()) + batch * seq)
        hidden = torch.rand((batch, seq, dim), generator=generator)
        return SimpleNamespace(last_hidden_state=hidden)

    return MagicMock(side_effect=forward)


def make_embedder(max_seq_length: int = 512) -> TorchEmbedder:
    info = ModelInfo(
        source_model_id="test/model",
        embedding_dim=DIM,
        max_seq_length=max_seq_length,
        serialization_format="safetensors",
        artifact_sha256="deadbeef",
    )
    return TorchEmbedder(
        model=make_model(),
        tokenizer=make_tokenizer(),
        info=info,
        max_seq_length=max_seq_length,
    )


def test_returns_one_unit_norm_vector_per_text():
    vectors = make_embedder().embed(["alpha beta", "gamma"])

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    for vector in vectors:
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_empty_input_short_circuits_without_touching_the_model():
    embedder = make_embedder()

    assert embedder.embed([]) == []
    embedder._model.assert_not_called()
    embedder._tokenizer.assert_not_called()


def test_tokenizer_is_asked_to_pad_and_truncate():
    """Without padding the batch is ragged; without truncation a long legal
    clause overruns the model's max_position_embeddings."""
    embedder = make_embedder(max_seq_length=128)

    embedder.embed(["alpha beta"])

    kwargs = embedder._tokenizer.call_args.kwargs
    assert kwargs["padding"] is True
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 128
    assert kwargs["return_tensors"] == "pt"


def test_long_input_is_truncated_to_max_seq_length():
    embedder = make_embedder(max_seq_length=16)

    embedder.embed(["alpha " * 500])

    assert embedder._model.call_args.kwargs["input_ids"].shape[1] <= 16


def test_pads_to_the_batch_longest_not_to_max_seq_length():
    embedder = make_embedder(max_seq_length=512)

    embedder.embed(["alpha", "beta gamma delta"])

    # The "query: " prefix contributes one token, so the rows are 2 and 4 long
    # and the batch pads to 4 -- not to max_seq_length.
    seq_len = embedder._model.call_args.kwargs["input_ids"].shape[1]
    assert seq_len == 4, f"expected batch-max padding, got {seq_len}"


def test_prefix_is_prepended_before_tokenization():
    embedder = make_embedder()

    embedder.embed(["alpha"], prefix="passage: ")

    assert embedder._tokenizer.call_args.args[0] == ["passage: alpha"]


def test_default_prefix_is_the_query_form():
    embedder = make_embedder()

    embedder.embed(["alpha"])

    assert embedder._tokenizer.call_args.args[0] == ["query: alpha"]


def test_the_tokenizers_attention_mask_reaches_the_pooling_step(
    monkeypatch: pytest.MonkeyPatch,
):
    """The mask must be forwarded, or padded positions would be averaged in and
    a short text's vector would shift depending on its batch-mates."""
    seen: dict[str, np.ndarray] = {}
    real_mean_pool = module_under_test.mean_pool

    def spy(last_hidden, attention_mask):
        seen["mask"] = attention_mask
        return real_mean_pool(last_hidden, attention_mask)

    monkeypatch.setattr(module_under_test, "mean_pool", spy)
    make_embedder().embed(["alpha", "beta gamma"])

    # "query: alpha" is 2 tokens and "query: beta gamma" is 3, so the shorter
    # row carries one padded position that the mask must zero out.
    np.testing.assert_array_equal(seen["mask"], np.array([[1, 1, 0], [1, 1, 1]]))


def test_runs_without_building_a_grad_graph():
    """Inference under grad would leak memory across requests."""
    grad_states: list[bool] = []

    def forward(input_ids=None, attention_mask=None, **_):
        grad_states.append(torch.is_grad_enabled())
        batch, seq = input_ids.shape
        return SimpleNamespace(last_hidden_state=torch.ones((batch, seq, DIM)))

    embedder = make_embedder()
    embedder._model = MagicMock(side_effect=forward)

    embedder.embed(["alpha"])

    assert grad_states == [False]


def test_info_is_exposed():
    info = make_embedder().info

    assert info.embedding_dim == DIM
    assert info.pooling == "mean"
    assert info.normalized is True
    assert info.serialization_format == "safetensors"


def test_from_settings_raises_when_weights_are_absent_and_download_is_off(
    tmp_path: Path,
):
    """An air-gapped deployment must fail loudly, not reach out to the network.

    `allow_download=False` is also what keeps this test offline: with the
    default of True it would download ~470 MB from Hugging Face.
    """
    settings = Settings(artifacts_dir=tmp_path, allow_download=False)

    with pytest.raises(FileNotFoundError, match="weights directory is missing"):
        TorchEmbedder.from_settings(settings)


def test_the_error_names_the_path_and_the_flag(tmp_path: Path):
    settings = Settings(artifacts_dir=tmp_path, allow_download=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        TorchEmbedder.from_settings(settings)

    message = str(excinfo.value)
    assert "ALLOW_DOWNLOAD" in message
    assert str(tmp_path) in message


def test_from_settings_prefers_baked_in_weights_over_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When weights are present locally, loading must be local_files_only.

    Otherwise a container with baked-in weights can still silently contact
    Hugging Face at startup, which breaks offline deployment and makes the
    image's provenance unverifiable.
    """
    (tmp_path / "model").mkdir()
    settings = Settings(artifacts_dir=tmp_path, allow_download=True)

    calls: list[dict] = []

    class StubAuto:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append({"source": str(source), **kwargs})
            stub = MagicMock()
            stub.config.hidden_size = DIM
            return stub

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", StubAuto)
    monkeypatch.setattr(transformers, "AutoModel", StubAuto)

    TorchEmbedder.from_settings(settings)

    assert calls, "from_pretrained was never called"
    for call in calls:
        assert call["local_files_only"] is True
        assert str(tmp_path) in call["source"]


def test_metadata_json_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Missing provenance must not block startup; it falls back to the config."""
    (tmp_path / "model").mkdir()
    settings = Settings(artifacts_dir=tmp_path, allow_download=False)

    class StubAuto:
        @staticmethod
        def from_pretrained(source, **kwargs):
            stub = MagicMock()
            stub.config.hidden_size = DIM
            return stub

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", StubAuto)
    monkeypatch.setattr(transformers, "AutoModel", StubAuto)

    embedder = TorchEmbedder.from_settings(settings)

    assert embedder.info.source_model_id == settings.source_model_id


def test_sha256_of_streams_a_file(tmp_path: Path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello world")

    digest = module_under_test.sha256_of(target, chunk_size=4)

    assert digest == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
