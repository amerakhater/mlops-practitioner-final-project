"""Shared fixtures.

Two client fixtures matter here:

``client``       an app whose embedder dependency is overridden with a
                 ``FakeEmbedder``, so every route can be exercised with no
                 model weights and no FAISS index anywhere on disk.
``bare_client``  an app with no embedder at all, which is how the 503
                 degraded-mode paths get covered.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prodllm.api.deps import get_optional_embedder, get_optional_engine
from prodllm.api.main import app
from prodllm.config import Settings, get_settings
from prodllm.embedding.fake import FakeEmbedder
from prodllm.rag import RagEngine


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointed at an empty tmp dir, so nothing touches the real artifacts."""
    monkeypatch.setenv("PRODLLM_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield Settings()
    get_settings.cache_clear()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


CHUNK_FIXTURES: list[dict] = [
    {
        "chunk_id": "art-147-1",
        "article_number": 147,
        "citation": "Egyptian Civil Code, Article 147",
        "text_ar": "العقد شريعة المتعاقدين",
        "text_ar_normalized": "العقد شريعه المتعاقدين",
        "text_en": "The contract makes the law of the parties.",
        "source_page": 16,
        "book": "BOOK I",
        "chapter": "CHAPTER I",
        "section": None,
        "parts_total": 1,
    },
    {
        "chunk_id": "art-238-1",
        "article_number": 238,
        "citation": "Egyptian Civil Code, Article 238",
        "text_ar": "إذا كان تصرف المدين بعوض",
        "text_ar_normalized": "اذا كان تصرف المدين بعوض",
        "text_en": "If the act by the debtor is for valuable consideration.",
        "source_page": 30,
        "book": "BOOK I",
        "chapter": "CHAPTER II",
        "section": None,
        "parts_total": 1,
    },
    {
        "chunk_id": "art-82-1",
        "article_number": 82,
        "citation": "Egyptian Civil Code, Article 82",
        "text_ar": "كل شيء مستقر بحيزه",
        "text_ar_normalized": "كل شيء مستقر بحيزه",
        "text_en": "Things which are fixed are immovables.",
        "source_page": 7,
        "book": None,
        "chapter": None,
        "section": "SECTION III",
        "parts_total": 1,
    },
]


@pytest.fixture
def chunks() -> list[dict]:
    return [dict(chunk) for chunk in CHUNK_FIXTURES]


@pytest.fixture
def engine(fake_embedder: FakeEmbedder, chunks: list[dict]) -> RagEngine:
    """A real RagEngine over a real FAISS index built from three chunks.

    ``min_score`` is relaxed to -1.0 because FakeEmbedder's vectors are
    hash-derived, so genuine similarity is meaningless here. The threshold
    itself is tested separately.
    """
    index = RagEngine.build_index(chunks, fake_embedder)
    return RagEngine(embedder=fake_embedder, index=index, top_k=3, min_score=-1.0)


@pytest.fixture
def client(fake_embedder: FakeEmbedder, engine: RagEngine) -> Iterator[TestClient]:
    """A client whose embedder and engine are test doubles.

    Overriding ``get_optional_embedder`` is enough to control ``get_embedder``
    too, because the strict dependency is defined in terms of the optional one
    and FastAPI applies overrides at every node of the dependency graph. The
    same holds for the engine pair.
    """
    app.dependency_overrides[get_optional_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_optional_engine] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def bare_client() -> Iterator[TestClient]:
    """A client with no embedder and no index: the 503 / degraded paths."""
    app.dependency_overrides[get_optional_embedder] = lambda: None
    app.dependency_overrides[get_optional_engine] = lambda: None
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def caplog_json(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog
