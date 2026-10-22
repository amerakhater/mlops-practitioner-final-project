"""The four endpoints named in the Definition of Done.

Covers the happy paths, the 422 validation surface, and the degraded 503 mode
that applies when startup could not load a model.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import prodllm

ARABIC_QUERY = "ما هي مدة التقادم في القانون المدني؟"
ARABIC_PASSAGES = ["عقد الإيجار", "الالتزامات التعاقدية", "المادة الأولى"]


# --------------------------------------------------------------------- /health


def test_health_reports_healthy_with_the_documents_indexed(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["documents_indexed"] == 3
    assert body["model_loaded"] is True
    assert body["version"] == prodllm.__version__
    assert body["uptime_seconds"] >= 0


def test_health_stays_200_but_degraded_without_a_model(bare_client: TestClient):
    """Liveness must not depend on inference, or the orchestrator crash-loops
    a container that could at least report why it is unhealthy."""
    response = bare_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["documents_indexed"] == 0


# ------------------------------------------------------------------- /metadata


def test_metadata_describes_the_served_model(client: TestClient):
    response = client.get("/metadata")

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_dim"] == 384
    assert body["pooling"] == "mean"
    assert body["normalized"] is True
    assert body["supported_prefixes"] == ["query: ", "passage: "]
    for key in (
        "source_model_id",
        "serialization_format",
        "artifact_sha256",
        "max_seq_length",
        "service_version",
        "max_batch_size",
        "max_text_chars",
    ):
        assert key in body, f"missing {key}"


def test_metadata_is_503_without_a_model(bare_client: TestClient):
    response = bare_client.get("/metadata")

    assert response.status_code == 503
    assert "detail" in response.json()


# -------------------------------------------------------------------- /predict


def test_predict_returns_a_unit_norm_vector(client: TestClient):
    response = client.post("/predict", json={"text": ARABIC_QUERY})

    assert response.status_code == 200
    body = response.json()
    assert body["dim"] == 384
    assert len(body["embedding"]) == 384
    assert np.isclose(np.linalg.norm(body["embedding"]), 1.0, atol=1e-5)
    assert body["latency_ms"] >= 0


def test_predict_is_deterministic(client: TestClient):
    first = client.post("/predict", json={"text": ARABIC_QUERY}).json()["embedding"]
    second = client.post("/predict", json={"text": ARABIC_QUERY}).json()["embedding"]

    assert first == second


def test_predict_accepts_the_passage_prefix_and_it_changes_the_vector(
    client: TestClient,
):
    as_query = client.post(
        "/predict", json={"text": ARABIC_QUERY, "prefix": "query: "}
    ).json()["embedding"]
    as_passage = client.post(
        "/predict", json={"text": ARABIC_QUERY, "prefix": "passage: "}
    ).json()["embedding"]

    assert as_query != as_passage


def test_predict_sets_a_correlation_header(client: TestClient):
    response = client.post("/predict", json={"text": ARABIC_QUERY})

    assert "x-request-id" in response.headers


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"text": ""}, "empty string"),
        ({}, "missing field"),
        ({"text": 123}, "wrong type"),
        ({"text": "x" * 9000}, "over the character limit"),
        ({"text": "ok", "prefix": "bogus: "}, "unsupported prefix"),
    ],
)
def test_predict_rejects_bad_input(client: TestClient, payload, reason):
    assert client.post("/predict", json=payload).status_code == 422, reason


def test_predict_is_503_without_a_model(bare_client: TestClient):
    response = bare_client.post("/predict", json={"text": ARABIC_QUERY})

    assert response.status_code == 503


# -------------------------------------------------------------- /predict/batch


def test_predict_batch_returns_one_vector_per_text(client: TestClient):
    response = client.post("/predict/batch", json={"texts": ARABIC_PASSAGES})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(ARABIC_PASSAGES)
    assert len(body["embeddings"]) == len(ARABIC_PASSAGES)
    assert all(len(v) == 384 for v in body["embeddings"])


def test_predict_batch_preserves_order(client: TestClient):
    """Each row must match the single-text result for the same input, or the
    caller cannot line embeddings up with their source documents."""
    batch = client.post("/predict/batch", json={"texts": ARABIC_PASSAGES}).json()

    for text, vector in zip(ARABIC_PASSAGES, batch["embeddings"], strict=True):
        single = client.post("/predict", json={"text": text}).json()["embedding"]
        assert vector == pytest.approx(single, abs=1e-6)


def test_predict_batch_handles_mixed_scripts(client: TestClient):
    response = client.post(
        "/predict/batch", json={"texts": ["عقد الإيجار", "Lease Agreement", "المادة 5"]}
    )

    assert response.status_code == 200
    assert response.json()["count"] == 3


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"texts": []}, "empty batch"),
        ({"texts": ["x"] * 65}, "over the batch limit"),
        ({"texts": ["ok", ""]}, "contains an empty string"),
        ({"texts": ["ok", "y" * 9000]}, "an item over the character limit"),
        ({}, "missing field"),
    ],
)
def test_predict_batch_rejects_bad_input(client: TestClient, payload, reason):
    assert client.post("/predict/batch", json=payload).status_code == 422, reason


def test_predict_batch_is_503_without_a_model(bare_client: TestClient):
    response = bare_client.post("/predict/batch", json={"texts": ARABIC_PASSAGES})

    assert response.status_code == 503


# --------------------------------------------------------------------- OpenAPI


def test_openapi_exposes_exactly_the_four_documented_routes(client: TestClient):
    paths = set(client.get("/openapi.json").json()["paths"])

    assert {"/health", "/metadata", "/predict", "/predict/batch"} <= paths


def test_openapi_version_tracks_the_package_version(client: TestClient):
    assert client.get("/openapi.json").json()["info"]["version"] == prodllm.__version__
