"""Column splitting, article-marker detection and heading capture."""

from prodllm.corpus.parse import (
    arabic_digits_to_int,
    classify_heading,
    find_repealed_ranges,
    group_lines,
    is_heading_title,
    iter_article_starts,
    ltr_line_text,
    match_arabic_article_marker,
    split_columns,
    strip_article_marker,
)


def word(text: str, x0: float, top: float = 100.0) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + 20, "top": top}


# ----------------------------------------------------------------- columns


def test_split_columns_is_positional_not_script_based():
    """Latin digits inside the Arabic column must stay in the Arabic column.

    Article numbers and statute references are printed in digits on the Arabic
    side; a "Latin text belongs to English" rule routes them to the wrong side.
    """
    words = [word("Article", 40.0), word("982", 400.0), word("ةدام", 440.0)]

    left, right = split_columns(words, gutter=292.0)

    assert [w["text"] for w in left] == ["Article"]
    assert [w["text"] for w in right] == ["982", "ةدام"]


def test_split_columns_uses_the_word_midpoint():
    # Spans the gutter but its midpoint (295) is to the right of it.
    straddling = {"text": "x", "x0": 285.0, "x1": 305.0, "top": 1.0}

    left, right = split_columns([straddling], gutter=292.0)

    assert not left
    assert len(right) == 1


# ------------------------------------------------------------------- lines


def test_group_lines_clusters_by_top_coordinate():
    words = [word("a", 10, 100.0), word("b", 30, 101.0), word("c", 10, 140.0)]

    lines = group_lines(words, tolerance=3.0)

    assert [len(line) for line in lines] == [2, 1]


def test_ltr_line_text_orders_by_x():
    assert ltr_line_text([word("world", 60), word("hello", 10)]) == "hello world"


# ---------------------------------------------------- english article marker


def test_detects_article_markers():
    lines = ["Article 1", "some body text", "Article 2"]

    assert list(iter_article_starts(lines)) == [(0, 1), (2, 2)]


def test_detects_a_marker_printed_without_a_space():
    """The source prints one marker as 'Article1022'; \\s+ silently drops it."""
    assert list(iter_article_starts(["Article1022"])) == [(0, 1022)]


def test_does_not_treat_a_mid_sentence_reference_as_a_marker():
    assert list(iter_article_starts(["The provisions of Article 982 apply"])) == []


def test_strip_article_marker_keeps_trailing_text():
    assert strip_article_marker("Article 5 The debtor may") == "The debtor may"
    assert strip_article_marker("Article 5") == ""


# ----------------------------------------------------- arabic article marker


def test_matches_a_plain_arabic_marker():
    assert match_arabic_article_marker("مادة ٢٣٨") == 238


def test_matches_a_marker_wrapped_in_brackets_by_bidi_reconstruction():
    assert match_arabic_article_marker("مادة ( (٨٢") == 82


def test_rejects_a_cross_reference_in_body_text():
    """Article 927 reads 'the provisions of Article 982 apply'.

    An unanchored search treats that as the start of article 982 and steals
    927's text, which is how three articles silently lost their Arabic.
    """
    body = "تسرى أحكام المادة ٩٨٢ فى أداء التعويض المنصوص عليه فى المواد"

    assert match_arabic_article_marker(body) is None


def test_rejects_a_long_line_that_merely_starts_with_the_marker_word():
    long_line = "مادة " + "ا" * 60

    assert match_arabic_article_marker(long_line) is None


def test_rejects_an_out_of_range_number():
    assert match_arabic_article_marker("مادة ٩٩٩٩") is None


def test_arabic_digits_to_int():
    assert arabic_digits_to_int("٢٣٨") == 238
    assert arabic_digits_to_int("238") == 238
    assert arabic_digits_to_int("abc") is None


# ---------------------------------------------------------------- headings


def test_classify_heading_recognises_capitalised_structural_labels():
    assert classify_heading("BOOK I") == ("book", "BOOK I")
    assert classify_heading("CHAPTER II") == ("chapter", "CHAPTER II")
    assert classify_heading("SECTION III") == ("section", "SECTION III")


def test_classify_heading_ignores_body_text_starting_with_a_keyword():
    assert classify_heading("Section 3 of the contract shall apply here") is None


def test_classify_heading_ignores_unrelated_lines():
    assert classify_heading("The debtor may") is None
    assert classify_heading("") is None


def test_is_heading_title_accepts_a_descriptive_title():
    assert is_heading_title("Obligations or Personal Rights") is True
    assert is_heading_title("THE CLASSIFICATION OF THINGS") is True


def test_is_heading_title_rejects_sentences():
    assert is_heading_title("The contract makes the law of the parties.") is False
    assert is_heading_title("(1) the debtor may") is False


# ----------------------------------------------------------------- repeals


def test_finds_repealed_ranges_in_both_phrasings():
    texts = [
        "* Articles 54-80 have been repealed by Presidential Decree.",
        "Articles 389 to 417 repealed",
    ]

    repealed = find_repealed_ranges(texts)

    assert 54 in repealed and 80 in repealed
    assert 389 in repealed and 417 in repealed
    assert len(repealed) == 27 + 29
    assert 53 not in repealed
    assert 81 not in repealed


def test_ignores_an_implausibly_wide_repeal_range():
    assert find_repealed_ranges(["Articles 1-9999 repealed"]) == set()
