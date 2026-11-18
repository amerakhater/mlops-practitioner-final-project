"""The sweepable chunking configuration: chunk_size, overlap, embedding_model.

The invariant these guard is the one the whole corpus design rests on: overlap
repeats text *inside* an article and never lets a chunk span two of them, so a
citation still points at exactly one article no matter how the sweep is tuned.
"""

from itertools import pairwise

import pytest

from prodllm.corpus.chunk import (
    DEFAULT_MAX_CHARS,
    ChunkConfig,
    _tail,
    chunk_corpus,
    chunk_corpus_with,
    split_by_length,
)
from prodllm.corpus.schema import Article


def make_article(number: int, text_ar: str = "نص المادة.", **kwargs) -> Article:
    return Article(article_number=number, text_ar=text_ar, source_page=1, **kwargs)


# ------------------------------------------------------------------ the config


def test_config_exposes_the_three_swept_parameters():
    params = ChunkConfig(name="baseline", chunk_size=800, overlap=120).as_params()

    assert params == {
        "config_name": "baseline",
        "chunk_size": 800,
        "overlap": 120,
        "embedding_model": "intfloat/multilingual-e5-small",
    }


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),  # a zero window packs nothing
        (-1, 0),
        (500, -1),  # negative overlap is meaningless
        (500, 500),  # each window would re-emit everything it consumed
        (500, 900),
    ],
)
def test_config_rejects_a_window_that_cannot_advance(chunk_size, overlap):
    with pytest.raises(ValueError):
        ChunkConfig(name="bad", chunk_size=chunk_size, overlap=overlap)


def test_config_is_hashable_so_it_can_key_a_run():
    assert len({ChunkConfig(name="a"), ChunkConfig(name="a")}) == 1


# ------------------------------------------------------------------- the tail


def test_tail_is_empty_when_overlap_is_disabled():
    assert _tail("one two three", 0) == ""


def test_tail_never_starts_mid_word():
    # A raw slice would yield "ngoing", a fragment that embeds as a different
    # token than the word it came from.
    assert _tail("the obligation is ongoing", 6) == "ongoing"


def test_tail_widens_rather_than_dropping_a_word_longer_than_the_overlap():
    # Trimming to the next space would return "" here and silently disable
    # the overlap at that boundary.
    assert _tail("alpha extraordinarily", 4) == "extraordinarily"


def test_tail_of_a_short_string_is_the_whole_string():
    assert _tail("short", 500) == "short"


# ----------------------------------------------------------- length splitting


def test_overlap_zero_leaves_the_pieces_disjoint():
    text = " ".join(f"sentence number {n}." for n in range(60))

    pieces = split_by_length(text, 200, overlap=0)

    assert len(pieces) > 1
    assert sum(len(p) for p in pieces) <= len(text) + len(pieces)


def test_each_piece_after_the_first_opens_with_the_previous_tail():
    text = " ".join(f"sentence number {n}." for n in range(60))

    pieces = split_by_length(text, 200, overlap=40)

    assert len(pieces) > 1
    for previous, piece in pairwise(pieces):
        assert piece.startswith(_tail(previous, 40))


def test_overlap_makes_the_pieces_cover_more_than_the_source():
    text = " ".join(f"sentence number {n}." for n in range(60))

    plain = split_by_length(text, 200, overlap=0)
    overlapped = split_by_length(text, 200, overlap=60)

    assert sum(len(p) for p in overlapped) > sum(len(p) for p in plain)


# --------------------------------------------------------- the article bound


def test_overlap_never_carries_text_across_an_article_boundary():
    first = "أ" * 40
    second = "ب" * 40
    articles = [make_article(1, first), make_article(2, second)]

    chunks = chunk_corpus(articles, max_chars=DEFAULT_MAX_CHARS, overlap=30)

    assert [c.article_number for c in chunks] == [1, 2]
    assert "ب" not in chunks[0].text_ar
    assert "أ" not in chunks[1].text_ar


def test_overlap_repeats_text_between_subclauses_of_one_article():
    # Short enough that each sub-clause clears max_chars on its own, which is
    # what routes the split through the sub-clause markers instead of length.
    text = "(١) " + "ألف " * 9 + "(٢) " + "باء " * 9

    plain = chunk_corpus([make_article(5, text)], max_chars=60, overlap=0)
    overlapped = chunk_corpus([make_article(5, text)], max_chars=60, overlap=20)

    assert len(plain) == len(overlapped) > 1
    assert all(c.article_number == 5 for c in overlapped)
    assert len(overlapped[1].text_ar) > len(plain[1].text_ar)


def test_chunk_corpus_with_matches_the_config_it_was_given():
    text = " ".join(f"جملة رقم {n}." for n in range(80))
    config = ChunkConfig(name="narrow", chunk_size=300, overlap=50)

    assert chunk_corpus_with([make_article(9, text)], config) == chunk_corpus(
        [make_article(9, text)], max_chars=300, overlap=50
    )


def test_the_default_config_reproduces_the_module_1_chunking():
    text = " ".join(f"جملة رقم {n}." for n in range(400))
    articles = [make_article(n, text) for n in (1, 2, 3)]

    assert chunk_corpus_with(articles, ChunkConfig(name="default")) == chunk_corpus(
        articles
    )
