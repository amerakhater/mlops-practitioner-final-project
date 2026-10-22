"""Correlation IDs and request logging, including the 404 and exception paths."""

import logging
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prodllm.api.middleware import CorrelationIdMiddleware


@pytest.fixture
def tiny_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    def boom() -> dict[str, bool]:
        raise RuntimeError("deliberate")

    return app


def test_request_id_header_is_a_uuid(tiny_app: FastAPI):
    with TestClient(tiny_app) as client:
        response = client.get("/ok")

    UUID(response.headers["x-request-id"])  # raises if malformed


def test_each_request_gets_a_distinct_id(tiny_app: FastAPI):
    with TestClient(tiny_app) as client:
        first = client.get("/ok").headers["x-request-id"]
        second = client.get("/ok").headers["x-request-id"]

    assert first != second


def test_header_matches_the_logged_correlation_id(
    tiny_app: FastAPI, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="prodllm.api")

    with TestClient(tiny_app) as client:
        response = client.get("/ok")

    completed = [r for r in caplog.records if r.message == "Request completed"]
    assert len(completed) == 1
    assert completed[0].correlation_id == response.headers["x-request-id"]


def test_completed_log_carries_the_request_fields(
    tiny_app: FastAPI, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="prodllm.api")

    with TestClient(tiny_app) as client:
        client.get("/ok")

    record = next(r for r in caplog.records if r.message == "Request completed")
    assert record.method == "GET"
    assert record.path == "/ok"
    assert record.status_code == 200
    assert record.latency_ms >= 0


def test_404s_are_still_instrumented(
    tiny_app: FastAPI, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="prodllm.api")

    with TestClient(tiny_app) as client:
        response = client.get("/nope")

    assert response.status_code == 404
    assert "x-request-id" in response.headers
    record = next(r for r in caplog.records if r.message == "Request completed")
    assert record.status_code == 404


def test_exceptions_are_logged_and_re_raised(
    tiny_app: FastAPI, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger="prodllm.api")

    with TestClient(tiny_app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    assert any(r.message == "Request failed" for r in caplog.records)
