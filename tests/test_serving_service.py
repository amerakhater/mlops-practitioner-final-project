"""The BentoML binding: routes, response shaping and status-code mapping.

``engine.py`` holds the behaviour and is tested in test_serving_core.py. What
is left here is the part only the decorators can get wrong -- a route that
moved, a streaming endpoint BentoML did not recognise as streaming, a field
dropped from the response, a ``ServiceUnavailable`` that would surface as a 500
instead of a 503.

The service is never started. ``@bentoml.service`` wraps the class and leaves
the original reachable as ``.inner``, so the methods are called directly
against a stub core. That keeps these tests as fast and as offline as the rest
of the suite -- no port, no model, no BentoML runtime.
"""

import asyncio

import pytest

import prodllm
from prodllm.rag import Answer, ReindexResult, RetrievedArticle
from prodllm.serving.engine import ServiceUnavailable

bentoml = pytest.importorskip(
    "bentoml",
    reason="the serving group is optional: `uv sync --group serving`",
)

from prodllm.serving.service import LegalRagService  # noqa: E402

SOURCE = RetrievedArticle(
    article_number=374,
    citation="Egyptian Civil Code, Article 374",
    text_ar="تتقادم الالتزامات بخمس عشرة سنة",
    text_en="Obligations are prescribed in fifteen years.",
    score=0.9049,
    source_page=57,
)


class StubCore:
    """Stands in for RagServiceCore, recording what the binding passed it."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple] = []
        self.settings = type("S", (), {"release_channel": "canary"})()

    def health(self) -> dict:
        return {"status": "healthy", "release_channel": "canary"}

    async def answer(self, question: str):
        if self.raises:
            raise self.raises
        self.calls.append(("answer", question))
        return Answer(answer="fifteen years", sources=[SOURCE], grounded=True), 12.5

    async def astream(self, question: str):
        if self.raises:
            raise self.raises
        self.calls.append(("astream", question))
        for fragment in ("fifteen ", "years"):
            yield fragment

    async def reindex(self, documents=None, persist=False, batch_size=16):
        if self.raises:
            raise self.raises
        self.calls.append(("reindex", documents, persist, batch_size))
        return ReindexResult(
            added=1, skipped=0, vectors_added=2, documents_indexed=1093
        )


def service(core: StubCore) -> LegalRagService.inner:
    """A service instance with the stub wired in, bypassing model loading."""
    instance = object.__new__(LegalRagService.inner)
    instance.core = core
    return instance


# --------------------------------------------------------------- route table


def test_the_four_endpoints_are_registered_at_their_documented_paths():
    routes = {name: api.route for name, api in LegalRagService.apis.items()}

    assert routes == {
        "ask": "/ask",
        "ask_stream": "/ask_stream",
        "reindex": "/reindex",
        "status": "/status",
    }


def test_ask_stream_is_recognised_as_streaming():
    """If BentoML does not see it as a stream it buffers the whole answer,
    which silently turns the streaming endpoint into a slower /ask."""
    assert LegalRagService.apis["ask_stream"].is_stream is True
    assert LegalRagService.apis["ask"].is_stream is False


def test_status_does_not_shadow_bentomls_own_health_probes():
    """BentoML serves /healthz, /livez and /readyz itself. Taking one of those
    names leaves a path where GET and POST give different answers."""
    routes = {api.route for api in LegalRagService.apis.values()}

    assert routes.isdisjoint({"/healthz", "/livez", "/readyz"})


def test_the_replica_advertises_more_concurrency_than_the_load_test_uses():
    """The 50-user run must measure service latency, not an admission limit."""
    assert LegalRagService.config["traffic"]["concurrency"] >= 50


def test_the_release_channel_is_published_as_a_label():
    assert LegalRagService.labels["release_channel"] == "stable"
    assert LegalRagService.labels["module"] == "3-serving"


# ------------------------------------------------------------ response shape


def test_ask_returns_the_answer_citations_and_release_identity():
    core = StubCore()

    body = asyncio.run(service(core).ask("ما هي مدة التقادم؟"))

    assert body["answer"] == "fifteen years"
    assert body["grounded"] is True
    assert body["latency_ms"] == 12.5
    assert body["release_channel"] == "canary"
    assert body["service_version"] == prodllm.__version__
    assert core.calls == [("answer", "ما هي مدة التقادم؟")]


def test_ask_shapes_each_source_through_the_shared_schema():
    """The citation shape is reused from prodllm.api.schemas rather than
    rebuilt, so the two serving surfaces cannot disagree about it."""
    body = asyncio.run(service(StubCore()).ask("q"))

    assert body["sources"] == [
        {
            "article_number": 374,
            "citation": "Egyptian Civil Code, Article 374",
            "source_page": 57,
            "score": 0.9049,
            # Null, not absent: this stub core retrieves vector-only, and the
            # field is how a client tells "no re-ranker ran" apart from "the
            # re-ranker scored it zero".
            "rerank_score": None,
            "text_ar": "تتقادم الالتزامات بخمس عشرة سنة",
            "text_en": "Obligations are prescribed in fifteen years.",
        }
    ]


def test_ask_stream_yields_the_fragments_unchanged():
    async def _drain() -> list[str]:
        return [f async for f in service(StubCore()).ask_stream("q")]

    assert asyncio.run(_drain()) == ["fifteen ", "years"]


def test_status_passes_the_cores_health_through():
    assert asyncio.run(_maybe(service(StubCore()).status())) == {
        "status": "healthy",
        "release_channel": "canary",
    }


async def _maybe(value):
    """status() is synchronous; this keeps the call style uniform."""
    return value


def test_reindex_returns_the_counts_and_forwards_its_arguments():
    core = StubCore()
    documents = [{"chunk_id": "art-999-1", "article_number": 999}]

    body = asyncio.run(service(core).reindex(documents, persist=True, batch_size=4))

    assert body == {
        "added": 1,
        "skipped": 0,
        "vectors_added": 2,
        "documents_indexed": 1093,
    }
    assert core.calls == [("reindex", documents, True, 4)]


# ------------------------------------------------------- status-code mapping


def test_ask_maps_a_missing_index_to_503():
    core = StubCore(raises=ServiceUnavailable("no index"))

    with pytest.raises(bentoml.exceptions.ServiceUnavailable) as caught:
        asyncio.run(service(core).ask("q"))

    assert caught.value.error_code.value == 503


def test_ask_stream_maps_a_missing_index_to_503():
    core = StubCore(raises=ServiceUnavailable("no index"))

    async def _drain() -> list[str]:
        return [f async for f in service(core).ask_stream("q")]

    with pytest.raises(bentoml.exceptions.ServiceUnavailable):
        asyncio.run(_drain())


def test_reindex_maps_a_missing_index_to_503():
    core = StubCore(raises=ServiceUnavailable("no index"))

    with pytest.raises(bentoml.exceptions.ServiceUnavailable):
        asyncio.run(service(core).reindex([]))


def test_reindex_maps_a_bad_batch_to_400_not_500():
    """An empty batch is the caller's mistake, so it must not look like ours."""
    core = StubCore(raises=ValueError("no documents supplied"))

    with pytest.raises(bentoml.exceptions.InvalidArgument) as caught:
        asyncio.run(service(core).reindex([]))

    assert caught.value.error_code.value == 400


def test_shutdown_closes_the_core():
    closed = {"value": False}

    class Closing(StubCore):
        async def aclose(self) -> None:
            closed["value"] = True

    asyncio.run(service(Closing())._shutdown())

    assert closed["value"] is True
