"""The artifact-free embedder used by the API tests and by bench_api --fake."""

import numpy as np

from prodllm.embedding.base import Embedder
from prodllm.embedding.fake import FakeEmbedder


def test_is_an_embedder():
    assert isinstance(FakeEmbedder(), Embedder)


def test_returns_one_unit_vector_per_text():
    vectors = FakeEmbedder().embed(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 384
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_is_deterministic():
    assert FakeEmbedder().embed(["alpha"]) == FakeEmbedder().embed(["alpha"])


def test_different_texts_give_different_vectors():
    first, second = FakeEmbedder().embed(["alpha", "beta"])

    assert first != second


def test_prefix_changes_the_vector():
    embedder = FakeEmbedder()

    as_query = embedder.embed(["alpha"], prefix="query: ")
    as_passage = embedder.embed(["alpha"], prefix="passage: ")

    assert as_query != as_passage


def test_batch_matches_element_wise_singles():
    embedder = FakeEmbedder()

    batch = embedder.embed(["alpha", "beta"])
    singles = [embedder.embed([t])[0] for t in ("alpha", "beta")]

    assert batch == singles


def test_empty_input_gives_empty_output():
    assert FakeEmbedder().embed([]) == []


def test_respects_a_custom_dimension():
    assert len(FakeEmbedder(dim=16).embed(["alpha"])[0]) == 16


def test_exposes_model_info():
    info = FakeEmbedder().info

    assert info.embedding_dim == 384
    assert info.pooling == "mean"
    assert info.normalized is True
