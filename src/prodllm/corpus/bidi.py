"""Reconstructing logical-order Arabic from a PDF's visual-order glyphs.

The source PDF stores glyphs in **visual** order, so an extractor hands back
Arabic words with their characters reversed: ``مادة`` arrives as ``ةدام``.

Reversing each line as a whole string looks like it works -- and it does, for
letters. It silently corrupts numbers. A number embedded in right-to-left text
is itself left-to-right, so its digits already arrive in logical order:
``٢٣٨`` extracts as digits 2, 3, 8. Reverse the string and article 238 becomes
article 832. Article 242 is a palindrome in Arabic-Indic digits and compares
equal either way, so spot-checking a single article does not catch this.

The fix is to reverse per word, and only for words that actually contain Arabic
letters.
"""

import re

# Arabic letters only -- deliberately excludes U+0660-U+0669 (Arabic-Indic
# digits), which must never be reversed.
_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")

# Reversing a string also reverses the direction brackets point in.
_MIRRORED = str.maketrans("()[]{}<>", ")(][}{><")


def has_arabic_letters(text: str) -> bool:
    """True if ``text`` contains at least one Arabic letter."""
    return _ARABIC_LETTER.search(text) is not None


def reverse_visual_arabic(word: str) -> str:
    """Restore logical order for one visually-ordered word.

    Words with no Arabic letters -- digit runs, Latin tokens, bare punctuation
    -- are returned unchanged, which is what keeps article numbers intact.
    """
    if not has_arabic_letters(word):
        return word
    return word[::-1].translate(_MIRRORED)


def rebuild_rtl_line(words: list[dict]) -> str:
    """Join one line of right-column words into logical-order Arabic text.

    Args:
        words: word dicts with at least ``x0``, as produced by a PDF
            extractor, all belonging to a single visual line.

    Words are ordered right-to-left by their left edge, then each is
    individually un-reversed.
    """
    ordered = sorted(words, key=lambda word: -word["x0"])
    return " ".join(reverse_visual_arabic(word["text"]) for word in ordered).strip()
