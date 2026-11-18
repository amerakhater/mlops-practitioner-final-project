"""The corpus record, article-level chunking, and the pre-embedding gate."""

import pytest

from prodllm.corpus.chunk import chunk_article, chunk_corpus, split_subclauses
from prodllm.corpus.schema import IMPLAUSIBLE_LENGTH, Article, Chunk
from prodllm.corpus.validate import validate_corpus


def make_article(number: int = 1, **overrides) -> Article:
    payload = {
        "article_number": number,
        "text_ar": "نص عربي",
        "text_en": "English text",
        "source_page": 1,
    }
    payload.update(overrides)
    return Article(**payload)


# ------------------------------------------------------------------ schema


def test_citation_is_derived_automatically():
    assert make_article(147).citation == "Egyptian Civil Code, Article 147"


def test_explicit_citation_is_respected():
    article = make_article(1, citation="Custom citation")

    assert article.citation == "Custom citation"


def test_normalized_arabic_is_populated_automatically():
    article = make_article(text_ar="الإستئناف")

    assert article.text_ar_normalized
    assert article.text_ar_normalized != article.text_ar


def test_accepts_the_ar_text_alias():
    """The handbook's field list says text_ar; its JSON example says ar_text."""
    article = Article(article_number=5, ar_text="نص", source_page=2)

    assert article.text_ar == "نص"


def test_serialises_under_the_text_ar_name():
    assert "text_ar" in make_article().model_dump(by_alias=True)


def test_is_empty_only_when_both_languages_are_blank():
    assert make_article(text_ar="", text_en="").is_empty is True
    assert make_article(text_ar="", text_en="x").is_empty is False


def test_detects_an_implausibly_long_record():
    assert make_article(text_ar="ا" * (IMPLAUSIBLE_LENGTH + 1)).is_implausibly_long


def test_article_number_must_be_positive():
    with pytest.raises(ValueError, match="article_number"):
        Article(article_number=0, source_page=1)


# ----------------------------------------------------------------- chunking


def test_a_short_article_becomes_one_chunk_carrying_its_citation():
    chunks = chunk_article(make_article(147))

    assert len(chunks) == 1
    assert chunks[0].article_number == 147
    assert chunks[0].citation == "Egyptian Civil Code, Article 147"
    assert chunks[0].parts_total == 1


def test_a_repealed_article_produces_no_chunks():
    """Embedding an empty record would let it surface as a nearest neighbour."""
    assert chunk_article(make_article(55, is_repealed=True, text_ar="")) == []


def test_an_article_with_no_arabic_produces_no_chunks():
    assert chunk_article(make_article(1022, text_ar="")) == []


def test_split_subclauses_splits_on_the_codes_own_numbering():
    text = "(١) الفقرة الأولى هنا (٢) الفقرة الثانية هنا"

    parts = split_subclauses(text)

    assert len(parts) == 2


def test_a_long_article_splits_into_citable_parts():
    long_text = "(١) " + "ا" * 900 + " (٢) " + "ب" * 900
    chunks = chunk_article(make_article(1143, text_ar=long_text), max_chars=1000)

    assert len(chunks) > 1
    # Every part must still be attributable to the article.
    assert {c.article_number for c in chunks} == {1143}
    assert all(str(1143) in c.citation for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.parts_total == len(chunks) for c in chunks)


def test_english_is_attached_only_to_the_first_part():
    """The two languages are not split in lockstep, so aligning them would lie."""
    long_text = "(١) " + "ا" * 900 + " (٢) " + "ب" * 900
    chunks = chunk_article(
        make_article(1, text_ar=long_text, text_en="English"), max_chars=1000
    )

    assert chunks[0].text_en == "English"
    assert all(c.text_en == "" for c in chunks[1:])


def test_a_very_long_unnumbered_article_still_splits():
    text = ". ".join("جملة طويلة جدا هنا" for _ in range(200))
    chunks = chunk_article(make_article(1, text_ar=text), max_chars=500)

    assert len(chunks) > 1
    assert all(len(c.text_ar) <= 700 for c in chunks)


def test_chunk_corpus_skips_repealed_and_keeps_the_rest():
    articles = [
        make_article(1),
        make_article(55, is_repealed=True, text_ar=""),
        make_article(3),
    ]

    chunks = chunk_corpus(articles)

    assert [c.article_number for c in chunks] == [1, 3]


def test_chunk_populates_normalized_text():
    chunk = Chunk(
        chunk_id="c1",
        article_number=1,
        text_ar="الإستئناف",
        citation="x",
        source_page=1,
    )

    assert chunk.text_ar_normalized


# --------------------------------------------------------------- validation


def test_a_clean_corpus_validates():
    report = validate_corpus([make_article(1), make_article(2)])

    assert report.ok
    assert report.errors == []
    assert report.total == 2
    assert (report.first_number, report.last_number) == (1, 2)


def test_an_unexplained_gap_is_an_error():
    report = validate_corpus([make_article(1), make_article(3)])

    assert not report.ok
    assert report.unexplained_gaps == [2]
    assert "unexplained gaps" in report.errors[0]


def test_a_repealed_article_is_kept_so_the_numbering_stays_contiguous():
    """Dropping a repealed article is indistinguishable from a parsing failure."""
    articles = [
        make_article(1),
        make_article(2, is_repealed=True, text_ar="", text_en=""),
        make_article(3),
    ]

    report = validate_corpus(articles)

    assert report.ok
    assert report.repealed == 1
    assert report.unexplained_gaps == []


def test_missing_arabic_is_an_error():
    report = validate_corpus([make_article(1, text_ar="")])

    assert not report.ok
    assert report.missing_arabic == [1]


def test_a_known_source_defect_is_tolerated_but_recorded():
    report = validate_corpus(
        [make_article(1022, text_ar="")], known_missing_arabic=frozenset({1022})
    )

    assert report.ok
    assert report.known_defects == [1022]
    assert report.missing_arabic == []


def test_a_new_defect_still_fails_even_with_an_allowlist():
    report = validate_corpus(
        [make_article(500, text_ar="")], known_missing_arabic=frozenset({1022})
    )

    assert not report.ok


def test_duplicate_article_numbers_are_an_error():
    report = validate_corpus([make_article(1), make_article(1)])

    assert not report.ok
    assert report.duplicates == [1]


def test_an_implausibly_long_record_is_an_error():
    report = validate_corpus([make_article(1, text_ar="ا" * 20_000)])

    assert not report.ok
    assert report.implausibly_long == [1]


def test_an_empty_corpus_reports_nothing_rather_than_crashing():
    report = validate_corpus([])

    assert report.total == 0
    assert report.ok
