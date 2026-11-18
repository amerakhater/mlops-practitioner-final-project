"""Log-level discipline (lesson 6, step 3).

The service uses levels deliberately: DEBUG for per-call latency, INFO for
lifecycle, WARNING for recoverable oddities, ERROR for model-load failure.
These tests pin that down so the levels do not drift into all-INFO noise.
"""

import logging

import pytest

from prodllm.instrumentation import log_latency


def test_log_latency_emits_at_debug(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)

    @log_latency("unit_work")
    def work() -> int:
        return 42

    assert work() == 42
    records = [r for r in caplog.records if r.message == "unit_work"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].latency_ms >= 0


def test_log_latency_is_silent_at_info(caplog: pytest.LogCaptureFixture):
    """This is what makes PRODLLM_LOG_LEVEL=DEBUG visibly change the output."""
    caplog.set_level(logging.INFO)

    @log_latency("unit_work")
    def work() -> int:
        return 42

    work()

    assert [r for r in caplog.records if r.message == "unit_work"] == []


def test_log_latency_preserves_the_wrapped_signature():
    """functools.wraps matters here: FastAPI introspects handler signatures."""

    @log_latency("documented")
    def annotated(a: int, b: str = "x") -> str:
        """Original docstring."""
        return f"{a}{b}"

    assert annotated.__name__ == "annotated"
    assert annotated.__doc__ == "Original docstring."
    assert annotated(1, b="y") == "1y"


def test_log_latency_logs_then_propagates_on_failure(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG)

    @log_latency("failing_work")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()

    assert any(r.message == "failing_work" for r in caplog.records)


def test_log_latency_accepts_a_custom_level(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)

    @log_latency("loud_work", level=logging.WARNING)
    def work() -> None:
        return None

    work()

    record = next(r for r in caplog.records if r.message == "loud_work")
    assert record.levelno == logging.WARNING
