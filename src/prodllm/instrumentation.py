"""A decorator for timing work and emitting it as a structured log line.

Written by hand rather than borrowed, because decorators are one of the module's
packaging-and-structure deliverables. It logs at DEBUG by default, which is what
makes ``PRODLLM_LOG_LEVEL=DEBUG`` visibly change the service's output, and it
rides the existing ``JsonFormatter`` so the ``extra`` keys are promoted to
top-level JSON fields rather than being buried in the message string.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger("prodllm.instrumentation")


def log_latency(
    event: str,
    level: int = logging.DEBUG,
    **static_fields: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Time the wrapped callable and log one line named ``event``.

    Args:
        event: the log message, used as a stable event name for log queries.
        level: the level to emit at. DEBUG keeps per-call noise out of INFO logs.
        static_fields: extra key/values attached to every emission.

    The timing is recorded in a ``finally`` block, so a failing call is still
    measured and logged before the exception propagates.
    """

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
                logger.log(
                    level,
                    event,
                    extra={"latency_ms": elapsed_ms, **static_fields},
                )

        return wrapper

    return decorate
