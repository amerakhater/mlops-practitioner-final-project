"""Arabic text normalization for retrieval.

Normalization is applied to a **copy** used for embedding and matching, never
to the text that gets shown or cited. A lawyer verifying an answer must see the
article as printed, diacritics and hamza forms intact; but a query written
"الاستئناف" should still match an article printed "الإستئناف". Storing both the
original and the normalized form is what lets retrieval be forgiving while
citation stays faithful.

The transformations, in order:

* strip tashkeel (fatha, damma, kasra, sukun, shadda, tanween) -- optional
  vowel marks that writers include inconsistently;
* strip tatweel (ـ), a purely typographic stretching character;
* unify alef variants أ إ آ ٱ -> ا, since hamza placement is widely
  inconsistent in practice;
* unify alef maksura ى -> ي and teh marbuta ة -> ه, the two most common
  orthographic splits in Arabic search;
* unify the waw and yeh hamza forms ؤ -> و and ئ -> ي;
* collapse runs of whitespace.

Note what is deliberately *not* done: no blanket ال -> لا substitution to
repair the source PDF's lam-alef ligature artifact. "ال" is also the Arabic
definite article, so such a rule would corrupt thousands of ordinary words.
"""

import re
import unicodedata

# U+064B..U+065F tashkeel, U+0670 superscript alef, U+06D6..U+06ED Quranic marks.
_TASHKEEL = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = re.compile(r"ـ")
_WHITESPACE = re.compile(r"\s+")

_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",  # أ alef with hamza above
        "إ": "ا",  # إ alef with hamza below
        "آ": "ا",  # آ alef with madda
        "ٱ": "ا",  # ٱ alef wasla
        "ى": "ي",  # ى alef maksura -> yeh
        "ة": "ه",  # ة teh marbuta -> heh
        "ؤ": "و",  # ؤ waw with hamza
        "ئ": "ي",  # ئ yeh with hamza
    }
)

# Arabic-Indic and extended Arabic-Indic digits -> ASCII.
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def strip_diacritics(text: str) -> str:
    """Remove tashkeel and tatweel, leaving letters and punctuation."""
    return _TATWEEL.sub("", _TASHKEEL.sub("", text))


def unify_letters(text: str) -> str:
    """Fold hamza, alef, alef-maksura and teh-marbuta variants."""
    return text.translate(_LETTER_MAP)


def arabic_digits_to_ascii(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII so numbers sort and parse."""
    return text.translate(_DIGIT_MAP)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    """Full normalization pipeline for the retrieval copy of a text."""
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = unify_letters(text)
    return collapse_whitespace(text)
