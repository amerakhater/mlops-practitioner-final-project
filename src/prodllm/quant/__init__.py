"""Quantization primitives: group-wise integer quantization and 4-bit packing.

Deliberately small and deliberately ours. The *checkpoint* produced in Module 4
is made by GPTQModel in a separate interpreter (see requirements-quant.txt);
what lives here is the arithmetic the project needs in its own process and its
own tests:

* ``scripts/quantization_analysis.py`` uses it to measure, on the real model,
  how much error each scheme introduces -- which is the evidence behind the
  report's claim that per-tensor symmetric INT8 is the wrong default.
* ``scripts/vram_table.py`` depends on the same 4.25-bits-per-weight accounting
  that ``quantization_bits_per_weight`` defines here, so the memory table and
  the error measurements cannot disagree about what "4-bit" means.

None of it is in the serving path.
"""

from prodllm.quant.pack import (
    DEFAULT_GROUP_SIZE,
    group_dequantize,
    group_quantize,
    pack_int4,
    quantization_bits_per_weight,
    quantize_dequantize,
    quantize_per_channel_symmetric,
    quantize_per_tensor_symmetric,
    unpack_int4,
)

__all__ = [
    "DEFAULT_GROUP_SIZE",
    "group_dequantize",
    "group_quantize",
    "pack_int4",
    "quantization_bits_per_weight",
    "quantize_dequantize",
    "quantize_per_channel_symmetric",
    "quantize_per_tensor_symmetric",
    "unpack_int4",
]
