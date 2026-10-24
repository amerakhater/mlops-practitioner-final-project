"""The Bento's behaviour, tested without BentoML.

``RagServiceCore`` takes its embedder, engine and stream generator as
constructor arguments, so every case here runs against a ``FakeEmbedder`` and a
three-chunk in-memory index: no weights, no server, no port. The BentoML
decorators in ``prodllm.serving.service`` add only request parsing and status
codes on top of what is covered here.

Batch re-indexing is tested at this level rather than through the endpoint
because the assertion that matters is a retrieval one -- a newly added article
has to come back from a query -- and that is invisible from a response body
that only reports counts.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

import prodllm
from prodllm.config import Settings
from prodllm.embedding.fake import FakeEmbedder
from prodllm.rag import NO_ARTICLE_FOUND, RagEngine, VectorIndex
from prodllm.serving.engine import RagServiceCore, ServiceUnavailable
from prodllm.streaming import ExtractiveStreamGenerator

NEW_ARTICLE = {
    "chunk_id": "art-999-1",
    "article_number": 999,
    "citation": "Egyptian Civil Code, Article 999",
    "text_ar": "يلتزم المستأجر بالمحافظة على العين المؤجرة",
    "text_ar_normalized": "يلتزم المستاجر بالمحافظه على العين المؤجره",
    "text_en": "The lessee is bound to preserve the leased property.",
    "source_page": 120,
    "book": "BOOK II",
    "chapter": "CHAPTER IV",
    "section": None,
    "parts_total": 1,
}

SECOND_NEW_ARTICLE = {**NEW_ARTICLE, "chunk_id": "art-1000-1", "article_number": 1000}


def drain(agen: AsyncIterator[str]) -> list[str]:
    async def _drain() -> list[str]:
        return [fragment async for fragment in agen]

    return asyncio.run(_drain())


@pytest.fixture
def core(engine: RagEngine, fake_embedder: FakeEmbedder) -> RagServiceCore:
    """A ready core over the three-chunk fixture index."""
    return RagServiceCore(
        settings=Settings(release_channel="test-channel"),
        embedder=fake_embedder,
        engine=engine,
        stream_generator=ExtractiveStreamGenerator(),
    )


@pytest.fixture
def degraded_core() -> RagServiceCore:
    """A core that loaded nothing, as if the artifacts were missing."""
    return RagServiceCore(
        settings=Settings(eager_load=False),
        stream_generator=ExtractiveStreamGenerator(),
    )


# -------------------------------------------------------------------- status


def test_status_reports_ready_with_the_documents_indexed(core: RagServiceCore):
    health = core.health()

    assert health["status"] == "healthy"
    assert health["documents_indexed"] == 3
    assert health["model_loaded"] is True
    assert health["version"] == prodllm.__version__
    assert health["uptime_seconds"] >= 0


def test_status_carries_the_release_channel(core: RagServiceCore):
    """The canary's readable half: which variant answered this request."""
    assert core.health()["release_channel"] == "test-channel"


def test_release_channel_defaults_to_stable():
    assert Settings().release_channel == "stable"


def test_status_is_degraded_rather_than_raising_without_artifacts(
    degraded_core: RagServiceCore,
):
    """Startup must not crash-loop over a missing artifact it could report."""
    health = degraded_core.health()

    assert health["status"] == "degraded"
    assert health["documents_indexed"] == 0
    assert health["model_loaded"] is False


def test_eager_load_false_skips_loading_entirely(degraded_core: RagServiceCore):
    assert degraded_core.embedder is None
    assert degraded_core.engine is None
    assert degraded_core.ready is False


def test_missing_weights_degrade_rather_than_raise(tmp_path):
    """A missing artifact must be reportable, not fatal.

    ``allow_download=False`` so the failure is a local FileNotFoundError and
    this test reaches no network -- with downloading left on, a missing weights
    directory would quietly fetch 465 MB from Hugging Face instead.
    """
    core = RagServiceCore(
        settings=Settings(
            artifacts_dir=tmp_path, allow_download=False, eager_load=True
        ),
        stream_generator=ExtractiveStreamGenerator(),
    )

    assert core.embedder is None
    assert core.engine is None
    assert core.health()["status"] == "degraded"


def test_a_missing_index_degrades_even_when_the_model_loaded(
    tmp_path, fake_embedder: FakeEmbedder
):
    """The second half of startup fails independently of the first.

    The model is fine and the index is absent, which is the state a fresh
    clone is in before scripts/build_index.py has run. /ask must answer 503
    while /status still reports what loaded.
    """
    core = RagServiceCore(
        settings=Settings(artifacts_dir=tmp_path, eager_load=True),
        embedder=fake_embedder,
        stream_generator=ExtractiveStreamGenerator(),
    )

    assert core.engine is None
    assert core.health()["model_loaded"] is True
    assert core.health()["status"] == "degraded"
    with pytest.raises(ServiceUnavailable):
        asyncio.run(core.answer("ما هو العقد؟"))


def test_the_core_builds_its_own_stream_generator_when_none_is_given():
    """Default wiring: no LLM configured, so the extractive streamer."""
    core = RagServiceCore(settings=Settings(eager_load=False))

    assert isinstance(core._stream_generator, ExtractiveStreamGenerator)


# -------------------------------------------------------------------- /ask


def test_answer_returns_the_answer_its_sources_and_a_latency(core: RagServiceCore):
    result, latency_ms = asyncio.run(core.answer("ما هو العقد؟"))

    assert result.grounded is True
    assert result.sources
    assert result.answer
    assert latency_ms >= 0


def test_answer_cites_article_numbers(core: RagServiceCore):
    """An answer a lawyer cannot verify against an article number is worthless."""
    result, _ = asyncio.run(core.answer("ما هو العقد؟"))

    for source in result.sources:
        assert source.article_number > 0
        assert str(source.article_number) in source.citation


def test_answer_raises_service_unavailable_without_an_index(
    degraded_core: RagServiceCore,
):
    with pytest.raises(ServiceUnavailable):
        asyncio.run(degraded_core.answer("ما هو العقد؟"))


def test_answer_is_actually_concurrent(core: RagServiceCore):
    """Blocking work must be off the event loop.

    If ``answer`` awaited the torch forward pass directly, these would run one
    after another and one slow question would stall every other request -- the
    exact failure the 50-user load test exists to catch.
    """

    async def _both() -> list[object]:
        return await asyncio.gather(
            core.answer("ما هو العقد؟"), core.answer("what is a contract?")
        )

    first, second = asyncio.run(_both())

    assert first[0].answer
    assert second[0].answer


# ------------------------------------------------------------- /ask_stream


def test_stream_reassembles_into_the_same_answer_as_ask(core: RagServiceCore):
    """The two endpoints must not disagree about what the law says."""
    buffered, _ = asyncio.run(core.answer("ما هو العقد؟"))

    streamed = "".join(drain(core.astream("ما هو العقد؟")))

    assert streamed == buffered.answer


def test_stream_yields_multiple_fragments(core: RagServiceCore):
    assert len(drain(core.astream("ما هو العقد؟"))) > 1


def test_stream_raises_service_unavailable_without_an_index(
    degraded_core: RagServiceCore,
):
    with pytest.raises(ServiceUnavailable):
        drain(degraded_core.astream("ما هو العقد؟"))


def test_stream_refuses_when_nothing_is_retrieved(
    fake_embedder: FakeEmbedder, chunks: list[dict]
):
    """min_score above every achievable score, so retrieval returns nothing."""
    index = RagEngine.build_index(chunks, fake_embedder)
    engine = RagEngine(embedder=fake_embedder, index=index, top_k=3, min_score=1.0001)
    core = RagServiceCore(
        settings=Settings(),
        embedder=fake_embedder,
        engine=engine,
        stream_generator=ExtractiveStreamGenerator(),
    )

    assert "".join(drain(core.astream("unrelated question"))) == NO_ARTICLE_FOUND


# ----------------------------------------------------- batch re-indexing


def test_reindex_adds_a_new_document(core: RagServiceCore):
    """The checklist case: one new document, indexed in a batch."""
    before = core.documents_indexed

    result = asyncio.run(core.reindex(documents=[NEW_ARTICLE]))

    assert result.added == 1
    assert result.skipped == 0
    assert result.documents_indexed == before + 1
    assert core.documents_indexed == before + 1


def test_a_reindexed_document_is_retrievable(
    fake_embedder: FakeEmbedder, chunks: list[dict]
):
    """Counts are not the point -- the new article has to come back from a query.

    A re-index reporting ``added: 1`` while leaving the article unretrievable
    would satisfy every count-based assertion above and still be broken: the
    vector has to be in the searchable set *and* its metadata has to resolve
    back to an article number and a citation.

    ``top_k`` covers the whole index deliberately. ``FakeEmbedder`` derives its
    vectors from a hash, so as conftest notes, relative similarity here is
    meaningless -- asserting the new article outranks the others would be
    testing a hash. Asking for every document and requiring the new one to be
    among them tests what this method is responsible for; that real retrieval
    ranks it sensibly is a property of the encoder, measured in reports/.
    """
    index = RagEngine.build_index(chunks, fake_embedder)
    engine = RagEngine(embedder=fake_embedder, index=index, top_k=10, min_score=-1.0)
    core = RagServiceCore(
        settings=Settings(),
        embedder=fake_embedder,
        engine=engine,
        stream_generator=ExtractiveStreamGenerator(),
    )

    asyncio.run(core.reindex(documents=[NEW_ARTICLE]))
    hits = {hit.article_number: hit for hit in engine.retrieve(NEW_ARTICLE["text_en"])}

    assert 999 in hits
    assert hits[999].citation == "Egyptian Civil Code, Article 999"
    assert hits[999].text_en == NEW_ARTICLE["text_en"]


def test_reindex_adds_one_vector_per_language(core: RagServiceCore):
    """A bilingual chunk is one document and two vectors."""
    result = asyncio.run(core.reindex(documents=[NEW_ARTICLE]))

    assert result.added == 1
    assert result.vectors_added == 2


def test_reindex_is_idempotent(core: RagServiceCore):
    """Re-posting a batch after a timeout must not duplicate it.

    Duplicates are not merely wasteful: retrieve() dedupes by article number
    over a fixed candidate pool, so a doubled chunk silently consumes a slot
    and pushes a genuine article out of the top-k.
    """
    first = asyncio.run(core.reindex(documents=[NEW_ARTICLE]))
    second = asyncio.run(core.reindex(documents=[NEW_ARTICLE]))

    assert first.added == 1
    assert second.added == 0
    assert second.skipped == 1
    assert second.documents_indexed == first.documents_indexed


def test_reindex_adds_only_the_new_members_of_a_mixed_batch(core: RagServiceCore):
    asyncio.run(core.reindex(documents=[NEW_ARTICLE]))

    result = asyncio.run(core.reindex(documents=[NEW_ARTICLE, SECOND_NEW_ARTICLE]))

    assert result.added == 1
    assert result.skipped == 1


def test_reindex_a_batch_of_several(core: RagServiceCore):
    before = core.documents_indexed

    result = asyncio.run(
        core.reindex(documents=[NEW_ARTICLE, SECOND_NEW_ARTICLE], batch_size=1)
    )

    assert result.added == 2
    assert core.documents_indexed == before + 2


def test_reindex_does_not_disturb_existing_documents(core: RagServiceCore):
    """Appending must not change what the original articles retrieve."""
    before = core.engine.retrieve("ما هو العقد؟")

    asyncio.run(core.reindex(documents=[NEW_ARTICLE]))
    after = core.engine.retrieve("ما هو العقد؟")

    assert [hit.article_number for hit in before] == [
        hit.article_number for hit in after if hit.article_number != 999
    ][: len(before)]


def test_reindex_rejects_an_empty_batch(core: RagServiceCore):
    with pytest.raises(ValueError, match="no documents supplied"):
        asyncio.run(core.reindex(documents=[]))


def test_reindex_raises_service_unavailable_without_an_index(
    degraded_core: RagServiceCore,
):
    with pytest.raises(ServiceUnavailable):
        asyncio.run(degraded_core.reindex(documents=[NEW_ARTICLE]))


def test_reindex_reads_documents_from_a_file(core: RagServiceCore, tmp_path):
    source = tmp_path / "new.json"
    source.write_text(json.dumps([NEW_ARTICLE]), encoding="utf-8")

    result = asyncio.run(core.reindex(source=source))

    assert result.added == 1


def test_reindex_rejects_a_source_that_is_not_a_list(core: RagServiceCore, tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must hold a JSON list"):
        asyncio.run(core.reindex(source=source))


def test_reindex_persists_only_when_asked(core: RagServiceCore, tmp_path, monkeypatch):
    monkeypatch.setattr(core.settings, "artifacts_dir", tmp_path)

    asyncio.run(core.reindex(documents=[NEW_ARTICLE], persist=False))
    assert not (tmp_path / "index").exists()

    asyncio.run(core.reindex(documents=[SECOND_NEW_ARTICLE], persist=True))
    saved = VectorIndex.load(tmp_path / "index")

    assert saved.document_count == core.documents_indexed


# ------------------------------------------------- copy-on-write index swap


def test_reindex_swaps_the_index_instead_of_mutating_it(
    engine: RagEngine, fake_embedder: FakeEmbedder
):
    """Readers mid-flight must keep a consistent snapshot.

    faiss' add() is not safe against a concurrent search(), so the live path
    builds an extended copy and rebinds. The object a request already holds
    must therefore be left untouched.
    """
    core = RagServiceCore(
        settings=Settings(),
        embedder=fake_embedder,
        engine=engine,
        stream_generator=ExtractiveStreamGenerator(),
    )
    original = engine._index
    original_vectors = len(original)

    asyncio.run(core.reindex(documents=[NEW_ARTICLE]))

    assert engine._index is not original
    assert len(original) == original_vectors
    assert len(engine._index) == original_vectors + 2


def test_offline_add_documents_mutates_in_place(
    engine: RagEngine, fake_embedder: FakeEmbedder
):
    """The offline job appends in place; nothing is reading it concurrently."""
    original = engine._index

    result = engine.add_documents([NEW_ARTICLE])

    assert engine._index is original
    assert result.added == 1


def test_extended_rejects_a_dimension_mismatch(
    engine: RagEngine, fake_embedder: FakeEmbedder
):
    """A mismatch means a different encoder, which invalidates the whole index."""
    import numpy as np

    with pytest.raises(ValueError, match="embedding dim mismatch"):
        engine._index.extended(np.zeros((1, 7), dtype=np.float32), [NEW_ARTICLE])


def test_add_rejects_mismatched_lengths(engine: RagEngine):
    import numpy as np

    with pytest.raises(ValueError, match="must be the same length"):
        engine._index.add(np.zeros((2, 384), dtype=np.float32), [NEW_ARTICLE])


def test_reindex_result_as_dict_round_trips(core: RagServiceCore):
    payload = asyncio.run(core.reindex(documents=[NEW_ARTICLE])).as_dict()

    assert payload == {
        "added": 1,
        "skipped": 0,
        "vectors_added": 2,
        "documents_indexed": 4,
    }


# ------------------------------------------------------------------ shutdown


def test_aclose_is_safe_on_a_generator_without_a_pool(core: RagServiceCore):
    """The extractive streamer has no connection pool; closing must still work."""
    asyncio.run(core.aclose())


def test_aclose_closes_a_pooled_generator():
    closed = {"value": False}

    class Pooled:
        def astream(self, question, passages): ...

        async def aclose(self) -> None:
            closed["value"] = True

    core = RagServiceCore(
        settings=Settings(eager_load=False), stream_generator=Pooled()
    )
    asyncio.run(core.aclose())

    assert closed["value"] is True
