"""The JSON formatter, the correlation-ID filter, and log-level discipline."""

import json
import logging

from prodllm.logging_conf import (
    CorrelationIdFilter,
    JsonFormatter,
    configure_logging,
    correlation_id_var,
)


def make_record(**kwargs) -> logging.LogRecord:
    defaults = {
        "name": "prodllm.test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "hello",
        "args": (),
        "exc_info": None,
    }
    defaults.update(kwargs)
    return logging.LogRecord(**defaults)


def test_format_emits_one_parseable_json_object():
    payload = json.loads(JsonFormatter().format(make_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "prodllm.test"
    assert payload["message"] == "hello"
    assert "timestamp" in payload
    assert "correlation_id" in payload


def test_extra_fields_are_promoted_to_top_level():
    record = make_record()
    record.latency_ms = 1.5
    record.path = "/predict"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["latency_ms"] == 1.5
    assert payload["path"] == "/predict"


def test_exception_info_becomes_an_exception_key():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record(exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_non_serializable_extras_fall_back_to_str():
    record = make_record()
    record.weird = object()

    payload = json.loads(JsonFormatter().format(record))

    assert isinstance(payload["weird"], str)


def test_correlation_id_filter_stamps_the_context_var():
    token = correlation_id_var.set("abc-123")
    record = make_record()
    try:
        assert CorrelationIdFilter().filter(record) is True
        assert record.correlation_id == "abc-123"
    finally:
        correlation_id_var.reset(token)


def test_configure_logging_replaces_handlers_and_sets_the_level():
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())

    configure_logging(level=logging.WARNING)

    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING

    configure_logging(level=logging.INFO)
