"""The shared benchmark harness: one timing loop, one hardware record.

Separate from ``prodllm.instrumentation``, which times the *served* request
path and emits a log line. This package is for offline measurement that
produces a comparison table, where the warmup, the repeat count and the
percentile have to be identical across the rows being compared.
"""

from prodllm.bench.hardware import GpuInfo, Hardware, describe_hardware, hardware_json
from prodllm.bench.timing import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    LatencyStats,
    measure,
)

__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUP",
    "GpuInfo",
    "Hardware",
    "LatencyStats",
    "describe_hardware",
    "hardware_json",
    "measure",
]
