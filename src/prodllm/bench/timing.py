"""One timing routine, used by every Module 4 measurement.

The module's requirement is a *consistent* benchmark harness, and the reason it
is a requirement is that optimization results are comparisons. Two numbers
produced by two different timing loops -- one that warmed up and one that did
not, one reporting a mean and one a median -- do not compare, and the
difference between them can easily exceed the speedup being claimed.

So the loop lives here and the callers supply only the callable:

* **Warmup is mandatory and discarded.** The first call through a torch model
  pays lazy kernel selection, allocator growth and page faults. On the
  distilled re-ranker the first forward pass measured ~6x the steady-state
  cost, which is larger than the speedup distillation bought -- so including
  it would have let a worse model look faster by being run first.
* **p50 and p95 rather than a mean.** A mean over a handful of repeats on a
  laptop is dominated by whatever else the OS was doing. The service's own
  latency numbers in reports/module-3.md are percentiles, so these match them.
* **perf_counter, not process_time.** The question is what a caller waits for,
  which includes time the thread spent descheduled.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 10


@dataclass(frozen=True)
class LatencyStats:
    """Wall-clock latency of a repeated call, in milliseconds."""

    repeats: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"p50 {self.p50_ms:.1f} ms, p95 {self.p95_ms:.1f} ms "
            f"(mean {self.mean_ms:.1f} +/- {self.stdev_ms:.1f}, n={self.repeats})"
        )


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than interpolated, because at n=10 an interpolated p95
    is a weighted average of the top two samples and reads as a measurement
    that was never taken. This returns a real observation.
    """
    if not ordered:
        raise ValueError("no samples")
    rank = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[rank]


def measure[R](
    call: Callable[[], R],
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> tuple[LatencyStats, R]:
    """Time ``call`` and return its statistics plus the last returned value.

    The value comes back so a caller can assert on it. A benchmark whose result
    is never inspected will happily time a function that raised nothing because
    it did nothing -- an empty candidate list re-ranks instantly.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    for _ in range(max(0, warmup)):
        call()

    samples: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000)

    ordered = sorted(samples)
    return (
        LatencyStats(
            repeats=repeats,
            warmup=max(0, warmup),
            mean_ms=statistics.fmean(samples),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            min_ms=ordered[0],
            max_ms=ordered[-1],
            stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        ),
        result,
    )
