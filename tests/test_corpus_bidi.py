"""Bidi reconstruction: the bug class that silently renumbers legal articles.

The source PDF stores glyphs in visual order. Reversing a whole line restores
the letters but destroys embedded numbers, because a number inside
right-to-left text is itself left-to-right. These tests pin the per-word rule
that gets both right.
"""

from prodllm.corpus.bidi import (
    has_arabic_letters,
    rebuild_rtl_line,
    reverse_visual_arabic,
)


def word(text: str, x0: float) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + 10, "top": 100.0}


def test_reverses_an_arabic_word():
    # "مادة" (article) is stored visually as "ةدام".
    assert reverse_visual_arabic("ةدام") == "مادة"


def test_leaves_arabic_indic_digits_untouched():
    """The whole point: article 238 must not become article 832."""
    assert reverse_visual_arabic("٢٣٨") == "٢٣٨"


def test_leaves_ascii_digits_and_latin_untouched():
    assert reverse_visual_arabic("238") == "238"
    assert reverse_visual_arabic("Article") == "Article"


def test_mirrors_brackets_when_reversing():
    assert reverse_visual_arabic("(ةدام") == "مادة)"


def test_has_arabic_letters_excludes_digit_only_tokens():
    assert has_arabic_letters("مادة") is True
    assert has_arabic_letters("٢٣٨") is False
    assert has_arabic_letters("238") is False
    assert has_arabic_letters("") is False


def test_rebuild_line_orders_right_to_left_and_keeps_the_number():
    # Visually: "٢٣٨" at the left, "ةدام" to its right.
    line = [word("٢٣٨", 300.0), word("ةدام", 340.0)]

    assert rebuild_rtl_line(line) == "مادة ٢٣٨"


def test_naive_whole_string_reversal_would_corrupt_the_number():
    """Documents the rejected alternative, so nobody 'simplifies' to it."""
    line = [word("٢٣٨", 300.0), word("ةدام", 340.0)]
    correct = rebuild_rtl_line(line)

    naive = " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))[::-1]

    assert correct == "مادة ٢٣٨"
    assert naive == "مادة ٨٣٢"
    assert correct != naive


def test_rebuild_line_handles_a_palindromic_number():
    """242 reads the same reversed, which is why one spot-check proves nothing."""
    line = [word("٢٤٢", 300.0), word("ةدام", 340.0)]

    assert rebuild_rtl_line(line) == "مادة ٢٤٢"


def test_rebuild_empty_line():
    assert rebuild_rtl_line([]) == ""
