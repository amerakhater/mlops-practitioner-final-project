"""Group-wise integer quantization, and a real packed layout for 4-bit weights.

Packing matters for honesty, not for speed. Module 4 asks for the optimized
model's size, and a "4-bit" checkpoint that stores each 4-bit value in its own
float32 is the same size as the original -- the saving is entirely in how the
numbers are laid out on disk. So the quantized artifact here stores two 4-bit
values per byte plus the per-group scales and zero points, and the size claimed
in reports/module-4.md is the size of that file as the filesystem reports it.

Nothing here runs a 4-bit *kernel*: dequantization is back to float for the
matmul. On this machine that is not a limitation being worked around, it is the
only option -- there is no reachable CUDA device (see
``prodllm.bench.hardware``). The consequence is stated wherever it matters:
quality numbers from this path are real, speed numbers are not.
"""

import torch

# The standard group size for 4-bit LLM weight quantization. One scale and one
# zero point per 128 consecutive input-dimension weights of an output channel.
DEFAULT_GROUP_SIZE = 128


def quantization_bits_per_weight(bits: int, group_size: int) -> float:
    """Effective stored bits per weight, including the group metadata.

    At 4 bits and group_size 128 this is 4.25, not 4.0: each group carries an
    FP16 scale and an FP16 zero point, so 32 extra bits are spread over 128
    weights. Ignoring that understates a 70B checkpoint by around 2 GiB, which
    is exactly the margin the "does it fit on one 80 GB card" question turns
    on.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return bits + 32.0 / group_size


def quantize_per_tensor_symmetric(weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """One scale for the whole tensor: the naive scheme, and the strawman.

    Symmetric means the zero point is pinned at zero, so the scale is set by
    the single largest magnitude anywhere in the tensor. One outlier weight
    therefore sets the resolution for every other weight in the matrix, which
    is the mechanism behind the error measurements in
    ``scripts/quantization_analysis.py``.
    """
    limit = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / limit
    if scale == 0:
        return weight.clone().float()
    return torch.clamp(torch.round(weight / scale), -limit - 1, limit) * scale


def quantize_per_channel_symmetric(weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """One scale per output channel: what any serious INT8 pipeline does.

    Still symmetric, but an outlier now only costs resolution within its own
    output channel instead of across the entire matrix. This is the cheapest
    possible fix to the per-tensor failure and it recovers most of the gap.
    """
    limit = 2 ** (bits - 1) - 1
    scale = weight.abs().amax(dim=1, keepdim=True) / limit
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.clamp(torch.round(weight / scale), -limit - 1, limit) * scale


def group_quantize(
    weight: torch.Tensor, bits: int = 4, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric group-wise quantization of a ``(out, in)`` weight matrix.

    Asymmetric -- a per-group zero point as well as a scale -- because at 4
    bits there are only 16 levels, so a group whose weights are not centred on
    zero would waste close to half of them under a symmetric scheme. At 8 bits
    the same choice is worth far less, which is part of why 4-bit schemes are
    asymmetric and group-wise almost without exception.

    Returns ``(codes, scales, zeros)`` where ``codes`` holds integers in
    ``[0, 2**bits - 1]`` with the same shape as ``weight``, and ``scales`` and
    ``zeros`` are ``(out, n_groups)``.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} is not a multiple of group_size "
            f"{group_size}; pad or choose a divisor"
        )

    levels = 2**bits - 1
    groups = weight.reshape(out_features, -1, group_size).float()
    low = groups.amin(dim=-1, keepdim=True)
    high = groups.amax(dim=-1, keepdim=True)
    scale = (high - low) / levels
    # A constant group (every weight identical) has zero range. Give it a unit
    # scale so the division is defined; every value maps to code 0 and
    # dequantizes back to the constant exactly.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round((groups - low) / scale), 0, levels)

    return (
        codes.reshape(out_features, in_features).to(torch.uint8),
        scale.squeeze(-1),
        low.squeeze(-1),
    )


def group_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Invert :func:`group_quantize`, returning float32."""
    out_features, in_features = codes.shape
    grouped = codes.reshape(out_features, -1, group_size).float()
    restored = grouped * scales.unsqueeze(-1) + zeros.unsqueeze(-1)
    return restored.reshape(out_features, in_features)


def quantize_dequantize(
    weight: torch.Tensor, bits: int = 4, group_size: int = DEFAULT_GROUP_SIZE
) -> torch.Tensor:
    """Round-trip a weight through quantization -- the "fake quant" operation.

    This is what GPTQ needs internally: it has to know what a column will
    become once quantized in order to compute the error it must compensate
    for in the columns it has not reached yet.
    """
    codes, scales, zeros = group_quantize(weight, bits, group_size)
    return group_dequantize(codes, scales, zeros, group_size)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two-per-byte along the last dimension.

    The low nibble holds the even-indexed value and the high nibble the odd
    one. ``in_features`` must be even, which it always is for a group size of
    128.
    """
    if codes.dtype != torch.uint8:
        raise ValueError(f"expected uint8 codes, got {codes.dtype}")
    if codes.max() > 15:
        raise ValueError("codes exceed 4 bits; pack_int4 needs values in [0, 15]")
    out_features, in_features = codes.shape
    if in_features % 2:
        raise ValueError(f"in_features {in_features} must be even to pack nibbles")
    pairs = codes.reshape(out_features, in_features // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Invert :func:`pack_int4`."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    stacked = torch.stack((low, high), dim=-1)
    return stacked.reshape(packed.shape[0], -1)[:, :in_features].contiguous()
