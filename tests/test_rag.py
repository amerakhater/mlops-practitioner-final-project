"""The RAG engine: ingestion, retrieval thresholds, and grounded generation."""

from pathlib import Path

import numpy as np
import pytest

from prodllm.config import Settings
from prodllm.embedding.fake import FakeEmbedder
from prodllm.rag import (
    Answer,
    ExtractiveGenerator,
    HttpGenerator,
    RagEngine,
    RetrievedArticle,
    VectorIndex,
    build_generator,
)


def article(number: int = 147, score: float = 0.9) -> RetrievedArticle:
    return RetrievedArticle(
        article_number=number,
        citation=f"Egyptian Civil Code, Article {number}",
        text_ar="العقد شريعة المتعاقدين",
        text_en="The contract makes the law of the parties.",
        score=score,
        source_page=16,
    )


# ------------------------------------------------------------ RetrievedArticle


def test_as_source_exposes_the_citation_and_not_internal_ids():
    payload = article().as_source()

    assert payload["article_number"] == 147
    assert payload["citation"] == "Egyptian Civil Code, Article 147"
    assert payload["source_page"] == 16
    assert "chunk_id" not in payload


def test_as_source_rounds_the_score():
    assert article(score=0.123456789).as_source()["score"] == 0.1235


# ------------------------------------------------------------------ generators


def test_extractive_generator_quotes_the_articles_with_citations():
    text = ExtractiveGenerator().generate("q", [article(147), article(238)])

    assert "Egyptian Civil Code, Article 147" in text
    assert "Egyptian Civil Code, Article 238" in text


def test_extractive_generator_declines_with_no_passages():
    text = ExtractiveGenerator().generate("q", [])

    assert "No article" in text


def test_extractive_generator_falls_back_to_arabic_when_english_is_absent():
    passage = RetrievedArticle(
        article_number=5,
        citation="Egyptian Civil Code, Article 5",
        text_ar="نص عربي",
        text_en="",
        score=0.9,
        source_page=1,
    )

    assert "نص عربي" in ExtractiveGenerator().generate("q", [passage])


def test_build_generator_defaults_to_extractive_without_an_llm():
    assert isinstance(build_generator(Settings()), ExtractiveGenerator)


def test_build_generator_returns_the_http_adapter_when_configured():
    settings = Settings(
        llm_base_url="https://example.invalid/v1",
        llm_model="some-model",
        llm_api_key="secret",
    )

    assert isinstance(build_generator(settings), HttpGenerator)


def test_llm_configured_requires_both_url_and_model():
    assert Settings(llm_base_url="https://x.invalid").llm_configured is False
    assert Settings(llm_model="m").llm_configured is False
    assert (
        Settings(llm_base_url="https://x.invalid", llm_model="m").llm_configured is True
    )


def test_http_generator_prompt_carries_citations_and_both_languages():
    generator = HttpGenerator("https://example.invalid/v1", "key", "model")

    prompt = generator.build_prompt("ما هو العقد؟", [article(147)])

    assert "Egyptian Civil Code, Article 147" in prompt
    assert "العقد شريعة المتعاقدين" in prompt
    assert "The contract makes the law" in prompt
    assert "ما هو العقد؟" in prompt


def test_http_generator_system_prompt_forbids_invention():
    prompt = HttpGenerator.SYSTEM_PROMPT.lower()

    assert "only" in prompt
    assert "never invent" in prompt
    assert "cite" in prompt


# ----------------------------------------------------------------- VectorIndex


def test_build_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        VectorIndex.build(np.zeros((2, 4), dtype=np.float32), [{"a": 1}])


def test_index_holds_one_vector_per_language_per_chunk(chunks):
    """Three bilingual chunks give six vectors but still three documents."""
    index = RagEngine.build_index(chunks, FakeEmbedder())

    assert len(index) == 6
    assert index.document_count == 3
    assert {entry["lang"] for entry in index.metadata} == {"ar", "en"}


def test_document_count_falls_back_to_article_number(chunks):
    for chunk in chunks:
        del chunk["chunk_id"]

    assert RagEngine.build_index(chunks, FakeEmbedder()).document_count == 3


def test_index_round_trips_through_disk(tmp_path: Path, chunks):
    original = RagEngine.build_index(chunks, FakeEmbedder())
    original.save(tmp_path / "index")

    reloaded = VectorIndex.load(tmp_path / "index")

    assert len(reloaded) == 6
    assert reloaded.document_count == 3
    assert reloaded.metadata[0]["article_number"] == 147

    query = np.asarray(FakeEmbedder().embed(["العقد"]), dtype=np.float32)
    assert original.search(query, 3) == reloaded.search(query, 3)


def test_load_raises_naming_the_missing_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="index artifact is missing"):
        VectorIndex.load(tmp_path / "nope")


def test_search_clamps_top_k_to_the_corpus_size(chunks):
    index = RagEngine.build_index(chunks, FakeEmbedder())
    query = np.asarray(FakeEmbedder().embed(["العقد"]), dtype=np.float32)

    assert len(index.search(query, top_k=50)) == 6


# ------------------------------------------------------------------- RagEngine


def test_chunk_payload_prefers_the_normalized_arabic():
    """Embedding the normalized copy is what makes hamza spellings match."""
    payload = RagEngine.chunk_payload(
        {"text_ar": "الإستئناف", "text_ar_normalized": "الاستئناف"}
    )

    assert payload == "الاستئناف"


def test_chunk_payload_normalizes_when_the_field_is_absent():
    # Both the alef-hamza and the yeh-hamza are folded: إ -> ا and ئ -> ي.
    assert RagEngine.chunk_payload({"text_ar": "الإستئناف"}) == "الاستيناف"


def test_chunk_payload_falls_back_to_english():
    assert RagEngine.chunk_payload({"text_ar": "", "text_en": "English"}) == "English"


def test_documents_indexed_counts_the_corpus(engine: RagEngine):
    assert engine.documents_indexed == 3


def test_retrieve_returns_citable_articles(engine: RagEngine):
    results = engine.retrieve("ما هو العقد؟")

    assert results
    assert all(isinstance(r, RetrievedArticle) for r in results)
    assert all(r.citation for r in results)


def test_retrieve_respects_top_k(fake_embedder: FakeEmbedder, chunks):
    index = RagEngine.build_index(chunks, fake_embedder)
    engine = RagEngine(fake_embedder, index, top_k=1, min_score=-1.0)

    assert len(engine.retrieve("ما هو العقد؟")) == 1


def test_a_high_threshold_filters_everything_out(fake_embedder: FakeEmbedder, chunks):
    """Returning nothing is correct for an unrelated question.

    A legal assistant that always surfaces its nearest neighbour invites the
    LLM to answer from an irrelevant article.
    """
    index = RagEngine.build_index(chunks, fake_embedder)
    engine = RagEngine(fake_embedder, index, min_score=1.01)

    assert engine.retrieve("entirely unrelated question") == []


def test_answer_declines_when_nothing_clears_the_threshold(
    fake_embedder: FakeEmbedder, chunks
):
    index = RagEngine.build_index(chunks, fake_embedder)
    engine = RagEngine(fake_embedder, index, min_score=1.01)

    result = engine.answer("entirely unrelated question")

    assert isinstance(result, Answer)
    assert result.grounded is False
    assert result.sources == []
    assert "No article" in result.answer


def test_answer_is_grounded_when_articles_are_found(engine: RagEngine):
    result = engine.answer("ما هو العقد؟")

    assert result.grounded is True
    assert result.sources
    assert result.answer


def test_answer_uses_the_injected_generator(engine: RagEngine, chunks, fake_embedder):
    class ShoutingGenerator:
        def generate(self, question, passages):
            return "SHOUTED"

    index = RagEngine.build_index(chunks, fake_embedder)
    custom = RagEngine(fake_embedder, index, ShoutingGenerator(), min_score=-1.0)

    assert custom.answer("q").answer == "SHOUTED"


def test_build_index_embeds_every_chunk_across_batches(fake_embedder, chunks):
    index = RagEngine.build_index(chunks * 5, fake_embedder, batch_size=2)

    assert len(index) == 30


def test_chunk_payloads_yields_both_languages(chunks):
    payloads = RagEngine.chunk_payloads(chunks[0])

    assert [lang for lang, _ in payloads] == ["ar", "en"]


def test_chunk_payloads_skips_a_missing_language(chunks):
    chunk = {**chunks[0], "text_en": ""}

    assert [lang for lang, _ in RagEngine.chunk_payloads(chunk)] == ["ar"]


def test_retrieve_returns_each_article_at_most_once(engine: RagEngine):
    """An article has an Arabic and an English vector; both must not be
    returned as if they were two separate sources."""
    results = engine.retrieve("ما هو العقد؟")

    numbers = [hit.article_number for hit in results]
    assert len(numbers) == len(set(numbers))


def test_build_index_on_an_empty_corpus_is_an_error(fake_embedder):
    """A zero-row index would make every question return nothing, silently."""
    with pytest.raises((ValueError, IndexError)):
        RagEngine.build_index([], fake_embedder)
