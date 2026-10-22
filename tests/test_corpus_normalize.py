"""Arabic normalization for the retrieval copy of each article."""

from prodllm.corpus.normalize import (
    arabic_digits_to_ascii,
    collapse_whitespace,
    normalize_arabic,
    strip_diacritics,
    unify_letters,
)


def test_strips_tashkeel():
    assert strip_diacritics("عَقْدٌ") == "عقد"


def test_strips_tatweel():
    assert strip_diacritics("عقــــد") == "عقد"


def test_unifies_alef_variants():
    assert unify_letters("أإآ") == "ااا"


def test_unifies_alef_maksura_and_teh_marbuta():
    assert unify_letters("ى") == "ي"
    assert unify_letters("ة") == "ه"


def test_unifies_hamza_carriers():
    assert unify_letters("ؤ") == "و"
    assert unify_letters("ئ") == "ي"


def test_arabic_indic_digits_to_ascii():
    assert arabic_digits_to_ascii("٢٣٨") == "238"
    assert arabic_digits_to_ascii("مادة ١٤٧") == "مادة 147"


def test_collapses_whitespace():
    assert collapse_whitespace("  a   b \n c ") == "a b c"


def test_normalize_makes_hamza_spellings_match():
    """A query spelled one way must match an article spelled the other."""
    assert normalize_arabic("الإستئناف") == normalize_arabic("الاستئناف")


def test_normalize_is_idempotent():
    once = normalize_arabic("إنَّ العَقْدَ شريعةُ المتعاقدين")

    assert normalize_arabic(once) == once


def test_normalize_preserves_digits():
    assert "٩٨٢" in normalize_arabic("أحكام المادة ٩٨٢")


def test_normalize_does_not_apply_a_lam_alef_repair():
    """A blanket ال -> لا rule would corrupt every definite article.

    The source PDF renders the lam-alef ligature as ال, but ال is also the
    Arabic definite article, so the substitution is not safely reversible and
    is deliberately not attempted.
    """
    assert normalize_arabic("القانون") == "القانون"
