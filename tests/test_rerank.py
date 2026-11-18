"""The second retrieval stage: the contract, the cross-encoder, the engine wiring."""

import logging
from pathlib import Path

import pytest
import torch

from prodllm.config import Settings
from prodllm.embedding.fake import FakeEmbedder
from prodllm.rag import (
    RagEngine,
    RetrievedArticle,
    build_reranker,
    rerank_payload,
)
from prodllm.rerank import CrossEncoderReranker, FakeReranker, Reranker, RerankerInfo


def article(number: int = 147, score: float = 0.9) -> RetrievedArticle:
    return RetrievedArticle(
        article_number=number,
        citation=f"Egyptian Civil Code, Article {number}",
        text_ar="العقد شريعة المتعاقدين",
        text_en="The contract makes the law of the parties.",
        score=score,
        source_page=16,
    )


class ScriptedReranker(Reranker):
    """Returns scores from a fixed list, so an ordering can be asserted exactly."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(model_id="scripted", parameters=0, max_seq_length=512)

    def score(self, query: str, passages) -> list[float]:
        self.calls.append((query, list(passages)))
        return self._scores[: len(passages)]


# ------------------------------------------------------------------ the contract


def test_rerank_returns_positions_best_first():
    ranked = ScriptedReranker([0.1, 0.9, 0.5]).rerank("q", ["a", "b", "c"])

    assert [position for position, _ in ranked] == [1, 2, 0]


def test_rerank_truncates_to_top_k():
    ranked = ScriptedReranker([0.1, 0.9, 0.5]).rerank("q", ["a", "b", "c"], top_k=2)

    assert [position for position, _ in ranked] == [1, 2]


def test_rerank_top_k_above_the_candidate_count_is_not_an_error():
    ranked = ScriptedReranker([0.1, 0.9]).rerank("q", ["a", "b"], top_k=50)

    assert len(ranked) == 2


def test_rerank_of_nothing_is_nothing_and_does_not_call_the_model():
    reranker = ScriptedReranker([])

    assert reranker.rerank("q", []) == []
    assert reranker.calls == []


def test_rerank_is_a_stable_sort_so_ties_keep_the_first_stage_order():
    """Equal scores must not reshuffle: the vector ordering is the tiebreak."""
    ranked = ScriptedReranker([0.5, 0.5, 0.5]).rerank("q", ["a", "b", "c"])

    assert [position for position, _ in ranked] == [0, 1, 2]


def test_rerank_rejects_a_score_count_that_does_not_match_the_passages():
    """A short score list would otherwise silently drop candidates."""
    with pytest.raises(ValueError, match="2 scores for 3 passages"):
        ScriptedReranker([0.1, 0.2]).rerank("q", ["a", "b", "c"])


# -------------------------------------------------------------------- the fake


def test_fake_reranker_is_deterministic_across_instances():
    first = FakeReranker().score("q", ["a", "b", "c"])
    second = FakeReranker().score("q", ["a", "b", "c"])

    assert first == second


def test_fake_reranker_scores_depend_on_the_query():
    assert FakeReranker().score("q1", ["a"]) != FakeReranker().score("q2", ["a"])


# ----------------------------------------------------------- the cross-encoder


class StubTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, first, second, **kwargs):
        self.calls.append({"first": first, "second": second, **kwargs})
        return {"input_ids": torch.zeros(len(first), 4, dtype=torch.long)}


class StubModel:
    """Returns one logit per row, shaped (batch, 1) as a regression head does."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, **encoded):
        rows = encoded["input_ids"].shape[0]
        self.batch_sizes.append(rows)
        logits = torch.arange(rows, dtype=torch.float32).reshape(-1, 1)
        return type("Output", (), {"logits": logits})()


def cross_encoder(
    batch_size: int = 8,
) -> tuple[CrossEncoderReranker, StubTokenizer, StubModel]:
    tokenizer, model = StubTokenizer(), StubModel()
    reranker = CrossEncoderReranker(
        model=model,
        tokenizer=tokenizer,
        info=RerankerInfo(model_id="stub", parameters=7, max_seq_length=128),
        batch_size=batch_size,
    )
    return reranker, tokenizer, model


def test_cross_encoder_scores_one_value_per_passage():
    reranker, _, _ = cross_encoder()

    assert len(reranker.score("q", ["a", "b", "c"])) == 3


def test_cross_encoder_passes_the_query_first_so_the_passage_gets_truncated():
    """`longest_first` truncation trims the longer side; the question must survive."""
    reranker, tokenizer, _ = cross_encoder()

    reranker.score("the question", ["passage one", "passage two"])

    call = tokenizer.calls[0]
    assert call["first"] == ["the question", "the question"]
    assert call["second"] == ["passage one", "passage two"]
    assert call["truncation"] is True
    assert call["max_length"] == 128


def test_cross_encoder_batches_according_to_batch_size():
    reranker, _, model = cross_encoder(batch_size=2)

    reranker.score("q", ["a", "b", "c", "d", "e"])

    assert model.batch_sizes == [2, 2, 1]


def test_cross_encoder_flattens_a_one_logit_head():
    reranker, _, _ = cross_encoder()

    assert reranker.score("q", ["a", "b"]) == [0.0, 1.0]


def test_cross_encoder_scores_nothing_without_a_forward_pass():
    reranker, _, model = cross_encoder()

    assert reranker.score("q", []) == []
    assert model.batch_sizes == []


# ------------------------------------------------------------- the pair payload


def test_rerank_payload_leads_with_the_citation_and_carries_both_languages():
    payload = rerank_payload(article(374))

    assert payload.startswith("[Egyptian Civil Code, Article 374]")
    assert "العقد شريعة المتعاقدين" in payload
    assert "The contract makes the law of the parties." in payload


def test_rerank_payload_omits_a_language_the_article_does_not_have():
    english_only = RetrievedArticle(
        article_number=1,
        citation="C",
        text_ar="",
        text_en="English only.",
        score=0.5,
        source_page=1,
    )

    assert rerank_payload(english_only) == "[C]\nEnglish only."


# ------------------------------------------------------------ engine integration


def two_stage(chunks: list[dict], reranker: Reranker, **kwargs) -> RagEngine:
    embedder = FakeEmbedder()
    return RagEngine(
        embedder=embedder,
        index=RagEngine.build_index(chunks, embedder),
        top_k=kwargs.pop("top_k", 2),
        min_score=-1.0,
        reranker=reranker,
        **kwargs,
    )


def test_engine_without_a_reranker_reports_none_and_leaves_scores_alone(
    engine: RagEngine,
):
    assert engine.reranker is None
    assert all(hit.rerank_score is None for hit in engine.retrieve("العقد"))


def test_engine_retrieval_order_follows_the_reranker_not_the_cosine(chunks: list[dict]):
    """The whole point of a second stage: it overrides the vector ordering."""
    embedder = FakeEmbedder()
    vector_only = RagEngine(
        embedder=embedder,
        index=RagEngine.build_index(chunks, embedder),
        top_k=3,
        min_score=-1.0,
    )
    baseline = [hit.article_number for hit in vector_only.retrieve("العقد")]

    # Score the baseline's last article highest, so a correct implementation
    # must promote it from the bottom of the list to the top.
    scores = [0.0] * len(baseline)
    scores[-1] = 1.0
    engine = two_stage(chunks, ScriptedReranker(scores), top_k=3)
    reranked = [hit.article_number for hit in engine.retrieve("العقد")]

    assert reranked[0] == baseline[-1]
    assert sorted(reranked) == sorted(baseline)


def test_engine_records_the_rerank_score_beside_the_cosine(chunks: list[dict]):
    engine = two_stage(chunks, ScriptedReranker([0.7, 0.2, 0.1]), top_k=2)

    hits = engine.retrieve("العقد")

    assert [hit.rerank_score for hit in hits] == [0.7, 0.2]
    # The cosine survives: the two are different measurements, and a client
    # reading `score` must not start receiving an unbounded logit instead.
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_engine_hands_the_reranker_the_raw_question_not_the_normalized_form(
    chunks: list[dict],
):
    """Normalization helps the embedder and throws away signal for a cross-encoder."""
    reranker = ScriptedReranker([0.5, 0.4, 0.3])
    two_stage(chunks, reranker, top_k=2).retrieve("العقد شريعة المتعاقدين؟")

    assert reranker.calls[0][0] == "العقد شريعة المتعاقدين؟"


def test_engine_shows_the_reranker_a_deeper_pool_than_it_returns(chunks: list[dict]):
    """It can only promote what it is shown, so the pool must exceed top_k."""
    reranker = ScriptedReranker([0.1] * 10)
    two_stage(chunks, reranker, top_k=1, rerank_candidates=3).retrieve("العقد")

    assert len(reranker.calls[0][1]) == 3


def test_engine_never_shows_the_reranker_fewer_candidates_than_top_k(
    chunks: list[dict],
):
    """rerank_candidates below top_k would silently shrink the answer."""
    engine = two_stage(
        chunks, ScriptedReranker([0.1] * 10), top_k=3, rerank_candidates=1
    )

    assert len(engine.retrieve("العقد")) == 3


def test_engine_falls_back_to_the_vector_order_when_the_reranker_raises(
    chunks: list[dict], caplog: pytest.LogCaptureFixture
):
    """A recoverable second-stage fault must not turn a good answer into a 500."""

    class Exploding(Reranker):
        @property
        def info(self) -> RerankerInfo:
            return RerankerInfo(model_id="boom", parameters=0, max_seq_length=512)

        def score(self, query, passages) -> list[float]:
            raise RuntimeError("cross-encoder unavailable")

    embedder = FakeEmbedder()
    index = RagEngine.build_index(chunks, embedder)
    vector_only = RagEngine(embedder=embedder, index=index, top_k=2, min_score=-1.0)
    expected = [hit.article_number for hit in vector_only.retrieve("العقد")]

    with caplog.at_level(logging.ERROR, logger="prodllm.rag"):
        degraded = two_stage(chunks, Exploding(), top_k=2).retrieve("العقد")

    assert [hit.article_number for hit in degraded] == expected
    assert all(hit.rerank_score is None for hit in degraded)
    assert "Re-ranking failed" in caplog.text


def test_engine_answers_from_the_reranked_passages(chunks: list[dict]):
    engine = two_stage(chunks, ScriptedReranker([0.0, 1.0, 0.0]), top_k=1)

    answer = engine.answer("العقد")

    assert answer.grounded
    assert answer.sources[0].citation in answer.answer


# ----------------------------------------------------------------- the factory


def test_build_reranker_returns_none_when_disabled():
    assert build_reranker(Settings(reranker_enabled=False)) is None


def test_build_reranker_degrades_rather_than_raising_when_the_artifact_is_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Vector-only is a working service, so a missing re-ranker is a warning."""
    settings = Settings(reranker_enabled=True, artifacts_dir=tmp_path)

    with caplog.at_level(logging.WARNING, logger="prodllm.rag"):
        assert build_reranker(settings) is None

    assert "no re-ranker artifact found" in caplog.text


def test_build_reranker_degrades_rather_than_raising_when_loading_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A directory that exists but holds no model must not fail startup."""
    (tmp_path / "reranker").mkdir()
    settings = Settings(reranker_enabled=True, artifacts_dir=tmp_path)

    with caplog.at_level(logging.WARNING, logger="prodllm.rag"):
        assert build_reranker(settings) is None

    assert "failed to load" in caplog.text


def test_reranker_source_prefers_an_explicit_model_id_over_the_directory():
    """So an A/B against the 568M teacher is config, not a code change."""
    settings = Settings(reranker_model_id="BAAI/bge-reranker-v2-m3")

    assert settings.reranker_source == "BAAI/bge-reranker-v2-m3"


def test_reranker_source_falls_back_to_the_artifacts_directory():
    settings = Settings(artifacts_dir=Path("arts"), reranker_dirname="rr")

    assert settings.reranker_source == str(Path("arts") / "rr")
