"""Group-wise quantization and 4-bit packing.

These get tested because reports/module-4.md quotes a compression ratio and a
bits-per-weight figure from them. A quantizer that silently loses a nibble
still produces plausible-looking numbers, and the round-trip assertions below
are the only thing that would notice.
"""

import pytest
import torch

from prodllm.quant import (
    group_dequantize,
    group_quantize,
    pack_int4,
    quantization_bits_per_weight,
    quantize_dequantize,
    quantize_per_channel_symmetric,
    quantize_per_tensor_symmetric,
    unpack_int4,
)


def test_four_bit_at_group_128_costs_4_25_bits_not_4():
    """The 0.25 is the FP16 scale and zero point spread over 128 weights."""
    assert quantization_bits_per_weight(4, 128) == pytest.approx(4.25)


def test_smaller_groups_cost_more_per_weight():
    """Finer groups quantize better and store more metadata -- the real trade."""
    assert quantization_bits_per_weight(4, 32) > quantization_bits_per_weight(4, 128)
    assert quantization_bits_per_weight(4, 32) == pytest.approx(5.0)


def test_group_size_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        quantization_bits_per_weight(4, 0)


# ------------------------------------------------------------ quantize/dequant


def test_quantize_produces_codes_inside_the_bit_width():
    weight = torch.randn(8, 128)

    codes, scales, zeros = group_quantize(weight, bits=4, group_size=128)

    assert codes.dtype == torch.uint8
    assert int(codes.max()) <= 15
    assert codes.shape == weight.shape
    assert scales.shape == (8, 1)
    assert zeros.shape == (8, 1)


def test_round_trip_is_close_and_bounded_by_half_a_step():
    """Asymmetric group quantization cannot err by more than scale/2."""
    torch.manual_seed(0)
    weight = torch.randn(4, 256)

    codes, scales, zeros = group_quantize(weight, bits=4, group_size=128)
    restored = group_dequantize(codes, scales, zeros, group_size=128)

    error = (weight - restored).abs()
    per_weight_limit = scales.repeat_interleave(128, dim=1) / 2 + 1e-6
    assert bool((error <= per_weight_limit).all())


def test_more_bits_strictly_reduces_error():
    torch.manual_seed(0)
    weight = torch.randn(4, 256)

    error_4 = (weight - quantize_dequantize(weight, 4, 128)).abs().mean()
    error_8 = (weight - quantize_dequantize(weight, 8, 128)).abs().mean()

    assert error_8 < error_4


def test_smaller_groups_strictly_reduce_error():
    """The other half of the trade that quantization_bits_per_weight prices."""
    torch.manual_seed(0)
    weight = torch.randn(4, 256)

    coarse = (weight - quantize_dequantize(weight, 4, 256)).abs().mean()
    fine = (weight - quantize_dequantize(weight, 4, 32)).abs().mean()

    assert fine < coarse


def test_a_constant_group_round_trips_exactly_and_does_not_divide_by_zero():
    """A zero-range group is the obvious NaN source; it must be exact instead."""
    weight = torch.full((2, 128), 0.75)

    restored = quantize_dequantize(weight, 4, 128)

    assert torch.allclose(restored, weight)
    assert not bool(restored.isnan().any())


def test_an_outlier_damages_its_own_group_and_not_the_others():
    """Why group-wise beats per-tensor: one huge value stays local.

    Asserted as a comparison against the tensor-wide alternative rather than
    as a fixed ratio between the two groups. The ratio is a property of the
    random draw -- it measured ~44x here -- whereas the claim the report makes
    is that the *clean* region survives group-wise quantization and does not
    survive a single tensor-wide scale, and that is what this checks.
    """
    torch.manual_seed(0)
    weight = torch.randn(1, 256) * 0.01
    weight[0, 0] = 100.0  # one huge value, in the first group only

    grouped = quantize_dequantize(weight, 4, 128)
    tensor_wide = quantize_dequantize(weight, 4, 256)

    clean = slice(128, 256)
    grouped_error = (weight[0, clean] - grouped[0, clean]).abs().mean()
    tensor_wide_error = (weight[0, clean] - tensor_wide[0, clean]).abs().mean()

    # The outlier's own group is hurt either way; the rest is rescued.
    assert grouped_error < tensor_wide_error / 10
    dirty = slice(0, 128)
    assert (weight[0, dirty] - grouped[0, dirty]).abs().mean() > grouped_error


def test_quantize_rejects_an_indivisible_in_features():
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        group_quantize(torch.randn(2, 130), bits=4, group_size=128)


def test_quantize_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="expected a 2-D weight"):
        group_quantize(torch.randn(2, 4, 8))


# --------------------------------------------------------- symmetric schemes


def test_per_channel_int8_beats_per_tensor_int8_at_the_same_bit_width():
    """The report's central INT8 claim: where the scales live is what matters.

    Same 8 bits either way. One outlier row is enough to separate them,
    because under a single tensor-wide scale that row sets the resolution for
    every other row in the matrix.
    """
    torch.manual_seed(0)
    weight = torch.randn(16, 64) * 0.02
    weight[0] *= 500  # one wildly-scaled output channel

    per_tensor = (weight - quantize_per_tensor_symmetric(weight, 8)).abs().mean()
    per_channel = (weight - quantize_per_channel_symmetric(weight, 8)).abs().mean()

    # Measured at 5.1x on this draw. Asserted loosely because the exact factor
    # is a property of the random matrix, not of the schemes.
    assert per_channel < per_tensor / 3


def test_per_tensor_int8_annihilates_the_non_outlier_rows_while_int4_groups_do_not():
    """Where the scales live beats how many bits they index.

    Measured on an outlier-heavy matrix, restricted to the rows that are *not*
    the outlier -- which is the population that matters, since the outlier is
    one channel out of thousands:

        per-tensor INT8  relative error on the clean rows: 1.00000
        group-wise INT4  relative error on the clean rows: 0.10334

    A relative error of 1.0 means those rows were quantized to zero outright:
    one channel 500x the others set a tensor-wide scale so coarse that every
    ordinary weight fell inside the first quantization step. Half as many bits,
    placed group-wise, keeps them. Note this is invisible in whole-matrix mean
    absolute error, where the outlier row dominates and per-tensor INT8 looks
    better -- which is exactly why the report measures relative error per role
    rather than one aggregate.
    """
    torch.manual_seed(0)
    weight = torch.randn(8, 256) * 0.02
    weight[0] *= 500
    clean = slice(1, 8)

    def relative_error_on_clean(restored: torch.Tensor) -> float:
        return float((weight[clean] - restored[clean]).norm() / weight[clean].norm())

    int8_per_tensor = relative_error_on_clean(quantize_per_tensor_symmetric(weight, 8))
    int4_grouped = relative_error_on_clean(quantize_dequantize(weight, 4, 128))

    assert int8_per_tensor > 0.99
    assert int4_grouped < 0.2
    assert int4_grouped < int8_per_tensor


def test_symmetric_schemes_keep_zero_at_zero():
    """Symmetric means the zero point is pinned; an exact zero must survive."""
    weight = torch.zeros(4, 32)
    weight[0, 0] = 1.0

    per_tensor = quantize_per_tensor_symmetric(weight, 8)
    per_channel = quantize_per_channel_symmetric(weight, 8)

    assert float(per_tensor[1, 5]) == 0.0
    assert float(per_channel[1, 5]) == 0.0


def test_an_all_zero_tensor_does_not_divide_by_zero():
    weight = torch.zeros(4, 32)

    assert not bool(quantize_per_tensor_symmetric(weight, 8).isnan().any())
    assert not bool(quantize_per_channel_symmetric(weight, 8).isnan().any())


def test_an_all_zero_channel_does_not_divide_by_zero():
    weight = torch.randn(4, 32)
    weight[2] = 0.0

    restored = quantize_per_channel_symmetric(weight, 8)

    assert not bool(restored.isnan().any())
    assert float(restored[2].abs().sum()) == 0.0


# ------------------------------------------------------------------- packing


def test_pack_halves_the_byte_count():
    codes = torch.randint(0, 16, (4, 128), dtype=torch.uint8)

    packed = pack_int4(codes)

    assert packed.shape == (4, 64)
    assert packed.numel() * packed.element_size() == codes.numel() // 2


def test_pack_unpack_round_trips_exactly():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (6, 256), dtype=torch.uint8)

    assert torch.equal(unpack_int4(pack_int4(codes), 256), codes)


def test_pack_keeps_the_even_value_in_the_low_nibble():
    """Stated in the docstring, so assert it rather than trust the comment."""
    codes = torch.tensor([[3, 10]], dtype=torch.uint8)

    packed = pack_int4(codes)

    assert int(packed[0, 0]) == 3 | (10 << 4)


def test_pack_rejects_values_that_do_not_fit_in_four_bits():
    codes = torch.tensor([[16, 0]], dtype=torch.uint8)

    with pytest.raises(ValueError, match="exceed 4 bits"):
        pack_int4(codes)


def test_pack_rejects_the_wrong_dtype():
    with pytest.raises(ValueError, match="expected uint8"):
        pack_int4(torch.zeros(2, 4, dtype=torch.int32))


def test_pack_rejects_an_odd_width():
    codes = torch.zeros(2, 5, dtype=torch.uint8)

    with pytest.raises(ValueError, match="must be even"):
        pack_int4(codes)


def test_full_pipeline_quantize_pack_unpack_dequantize_matches_fake_quant():
    """The stored artifact must decode to exactly what quantize_dequantize gives."""
    torch.manual_seed(0)
    weight = torch.randn(8, 256)

    codes, scales, zeros = group_quantize(weight, 4, 128)
    recovered = unpack_int4(pack_int4(codes), 256)
    through_disk = group_dequantize(recovered, scales, zeros, 128)

    assert torch.allclose(through_disk, quantize_dequantize(weight, 4, 128))
