"""The pooling maths, tested on hand-built arrays with no model involved.

This is where the e5 contract is pinned down: mean-over-tokens (NOT the CLS
token) honouring the attention mask, then L2 normalization.
"""

import numpy as np
import pytest

from prodllm.pooling import l2_normalize, mean_pool


def test_mean_pool_ignores_padded_positions():
    # Two sequences of 3 tokens, 4 hidden dims. Row 0 has 2 real tokens,
    # row 1 has 1. The padded positions hold deliberately extreme values:
    # if the mask were ignored they would dominate the mean.
    hidden = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0], [999.0, 999.0, 999.0, 999.0]],
            [[2.0, 4.0, 6.0, 8.0], [-999.0, -999.0, -999.0, -999.0], [999.0] * 4],
        ],
        dtype=np.float32,
    )
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64)

    pooled = mean_pool(hidden, mask)

    assert pooled.shape == (2, 4)
    np.testing.assert_allclose(pooled[0], [2.0, 2.0, 2.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(pooled[1], [2.0, 4.0, 6.0, 8.0], rtol=1e-6)


def test_mean_pool_with_all_ones_mask_is_a_plain_mean():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(3, 5, 8)).astype(np.float32)
    mask = np.ones((3, 5), dtype=np.int64)

    np.testing.assert_allclose(mean_pool(hidden, mask), hidden.mean(axis=1), rtol=1e-5)


def test_mean_pool_does_not_divide_by_zero_on_empty_mask():
    hidden = np.ones((1, 4, 3), dtype=np.float32)
    mask = np.zeros((1, 4), dtype=np.int64)

    pooled = mean_pool(hidden, mask)

    assert np.all(np.isfinite(pooled)), "an all-padding row must not produce nan/inf"


def test_mean_pool_is_not_cls_pooling():
    """Guard against silently switching to the CLS token, which e5 does not use."""
    hidden = np.array([[[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 1]], dtype=np.int64)

    pooled = mean_pool(hidden, mask)

    np.testing.assert_allclose(pooled[0], [10.0 / 3.0, 0.0], rtol=1e-6)


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]], dtype=np.float32)

    normalized = l2_normalize(vectors)

    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=1), np.ones(3), rtol=1e-6
    )
    np.testing.assert_allclose(normalized[0], [0.6, 0.8], rtol=1e-6)


def test_l2_normalize_survives_a_zero_vector():
    normalized = l2_normalize(np.zeros((1, 4), dtype=np.float32))

    assert np.all(np.isfinite(normalized))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pooling_keeps_shapes_across_dtypes(dtype):
    hidden = np.ones((2, 3, 6), dtype=dtype)
    mask = np.ones((2, 3), dtype=np.int64)

    assert l2_normalize(mean_pool(hidden, mask)).shape == (2, 6)
