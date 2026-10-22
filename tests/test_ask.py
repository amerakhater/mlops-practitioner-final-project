"""POST /ask -- the track's contract: an answer plus verifiable citations."""

import pytest
from fastapi.testclient import TestClient

ARABIC_QUESTION = "ما هي مدة التقادم في القانون المدني؟"


def test_ask_returns_an_answer_and_sources(client: TestClient):
    response = client.post("/ask", json={"question": ARABIC_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]
    assert body["grounded"] is True
    assert body["latency_ms"] >= 0


def test_sources_carry_article_citations_not_chunk_ids(client: TestClient):
    """An answer a lawyer cannot trace to an article number is worthless."""
    sources = client.post("/ask", json={"question": ARABIC_QUESTION}).json()["sources"]

    for source in sources:
        assert isinstance(source["article_number"], int)
        assert source["citation"].startswith("Egyptian Civil Code, Article")
        assert str(source["article_number"]) in source["citation"]
        assert source["source_page"] >= 1
        assert "chunk_id" not in source


def test_sources_are_ordered_by_descending_score(client: TestClient):
    sources = client.post("/ask", json={"question": ARABIC_QUESTION}).json()["sources"]

    scores = [source["score"] for source in sources]
    assert scores == sorted(scores, reverse=True)


def test_ask_respects_top_k(client: TestClient):
    sources = client.post("/ask", json={"question": ARABIC_QUESTION}).json()["sources"]

    assert len(sources) <= 3


def test_ask_works_for_an_english_question(client: TestClient):
    response = client.post("/ask", json={"question": "What makes a contract binding?"})

    assert response.status_code == 200
    assert response.json()["sources"]


def test_ask_is_deterministic(client: TestClient):
    first = client.post("/ask", json={"question": ARABIC_QUESTION}).json()
    second = client.post("/ask", json={"question": ARABIC_QUESTION}).json()

    assert first["answer"] == second["answer"]
    assert [s["article_number"] for s in first["sources"]] == [
        s["article_number"] for s in second["sources"]
    ]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"question": ""}, "empty question must be 422"),
        ({"question": "   "}, "whitespace-only question must be 422"),
        ({}, "missing field"),
        ({"question": 123}, "wrong type"),
        ({"question": "x" * 2100}, "over the character limit"),
    ],
)
def test_ask_rejects_bad_input(client: TestClient, payload, reason):
    assert client.post("/ask", json=payload).status_code == 422, reason


def test_ask_is_503_without_an_index(bare_client: TestClient):
    response = bare_client.post("/ask", json={"question": ARABIC_QUESTION})

    assert response.status_code == 503
    assert "detail" in response.json()


def test_ask_sets_a_correlation_header(client: TestClient):
    response = client.post("/ask", json={"question": ARABIC_QUESTION})

    assert "x-request-id" in response.headers


def test_openapi_documents_ask(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/ask" in paths
    assert "post" in paths["/ask"]
