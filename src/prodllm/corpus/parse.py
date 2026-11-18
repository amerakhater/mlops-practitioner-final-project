"""Turning extracted PDF words into article records.

Pure functions over word dictionaries, so the whole pipeline is testable with
hand-built input and no PDF. A word dict needs ``text``, ``x0``, ``x1`` and
``top``, which is what every PDF extractor provides.

The layout is two columns: English on the left, Arabic on the right. The split
is **positional**, not script-based, and that matters: article numbers, dates
and statute references appear as Latin or Arabic-Indic digits *inside* the
Arabic column, so a rule like "Latin text belongs to the English column" routes
them to the wrong side.
"""

import re
from collections.abc import Iterable, Iterator

from prodllm.corpus.bidi import rebuild_rtl_line

# "Article 238" in the English column. The authoritative marker: it is
# left-to-right, so it needs no bidi reconstruction to be read reliably.
# The space is optional because the source prints at least one marker as
# "Article1022", which a \s+ pattern silently drops.
ARTICLE_EN = re.compile(r"^\s*Article\s*(\d+)\b")

# An Arabic article heading: the bare word مادة followed by its number, on a
# line of its own.
#
# Anchoring at the line start is what distinguishes a heading from a
# cross-reference. Article 927's body reads "تسرى أحكام المادة ٩٨٢" -- "the
# provisions of Article 982 apply" -- and an unanchored search for مادة treats
# that as the start of article 982, silently stealing 927's text. The definite
# article ("المادة") only ever appears mid-sentence, so ^مادة excludes it.
# The separator is deliberately permissive. Per-word bidi reconstruction can
# leave the number wrapped in brackets with spaces between them, so the marker
# arrives as "مادة ( (٨٢" as often as plain "مادة ٨٢".
ARTICLE_AR = re.compile(r"^\s*مادة[\s\(\)]*([٠-٩0-9]+)")

# Heading lines carry only the marker; body text that begins with a citation
# would be far longer.
ARTICLE_AR_MAX_LEN = 30

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# "Articles 54-80 have been repealed ...", "Articles 389 to 417 repealed".
REPEAL_RANGE = re.compile(
    r"Articles?\s+(\d+)\s*(?:-|–|to)\s*(\d+)[^.\n]{0,80}?repealed", re.IGNORECASE
)

# Structural headings are printed in capitals in the English column.
HEADING_KEYWORDS = ("BOOK", "PART", "CHAPTER", "SECTION", "TITLE")
HEADING_LEVEL = {
    "BOOK": "book",
    "PART": "book",
    "TITLE": "chapter",
    "CHAPTER": "chapter",
    "SECTION": "section",
}


def split_columns(
    words: Iterable[dict], gutter: float
) -> tuple[list[dict], list[dict]]:
    """Partition words into (left/English, right/Arabic) by horizontal midpoint."""
    left: list[dict] = []
    right: list[dict] = []
    for word in words:
        midpoint = (word["x0"] + word["x1"]) / 2
        (right if midpoint >= gutter else left).append(word)
    return left, right


def group_lines(words: Iterable[dict], tolerance: float = 3.0) -> list[list[dict]]:
    """Cluster words into visual lines by their top coordinate.

    Clusters sequentially against the first word of the current line rather
    than bucketing on ``round(top / tolerance)``. Fixed buckets split a line
    whenever its words straddle a bucket boundary -- two words at top 100 and
    101 would land in different lines for no reason other than where the
    boundary happens to fall.
    """
    ordered = sorted(words, key=lambda word: word["top"])
    lines: list[list[dict]] = []
    for word in ordered:
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def ltr_line_text(words: Iterable[dict]) -> str:
    """Join left-column words in reading order."""
    return " ".join(
        word["text"] for word in sorted(words, key=lambda word: word["x0"])
    ).strip()


def classify_heading(line: str) -> tuple[str, str] | None:
    """Recognise a structural heading, returning ``(level, text)``.

    Headings are set in capitals, so a line with lower-case words is body text
    even when it happens to start with one of the keywords.
    """
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 90:
        return None
    first = stripped.split()[0].upper().strip(":.")
    if first not in HEADING_KEYWORDS:
        return None
    letters = [character for character in stripped if character.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.85:
        return None
    return HEADING_LEVEL[first], stripped


def arabic_digits_to_int(digits: str) -> int | None:
    """Parse an Arabic-Indic or ASCII digit run into a sortable integer."""
    try:
        return int(digits.translate(_ARABIC_INDIC))
    except ValueError:
        return None


def match_arabic_article_marker(line: str, max_number: int = 1200) -> int | None:
    """Return the article number if ``line`` is an Arabic article heading.

    Returns ``None`` for body text, including text that cites another article.
    """
    stripped = line.strip()
    if len(stripped) > ARTICLE_AR_MAX_LEN:
        return None
    match = ARTICLE_AR.match(stripped)
    if match is None:
        return None
    number = arabic_digits_to_int(match.group(1))
    if number is None or not 1 <= number <= max_number:
        return None
    return number


def is_heading_title(line: str) -> bool:
    """Recognise the descriptive title printed under a heading label.

    "BOOK I" on its own says nothing useful; the line beneath it carries
    "Obligations or Personal Rights". Such a line is short, has no sentence
    punctuation, and is set in capitals or title case.
    """
    stripped = line.strip()
    if not 3 <= len(stripped) <= 80:
        return False
    if stripped.endswith((".", ",", ";", ":")) or stripped.startswith("("):
        return False
    words = [w for w in re.split(r"\s+", stripped) if any(c.isalpha() for c in w)]
    if not words or len(words) > 10:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) >= 0.85:
        return True
    # Title Case: most words capitalised, ignoring short connectives.
    significant = [w for w in words if len(w) > 3]
    if not significant:
        return False
    return sum(w[0].isupper() for w in significant) / len(significant) >= 0.75


def find_repealed_ranges(page_texts: Iterable[str]) -> set[int]:
    """Collect every article number the document declares repealed."""
    repealed: set[int] = set()
    for text in page_texts:
        for match in REPEAL_RANGE.finditer(text):
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and end - start < 500:
                repealed.update(range(start, end + 1))
    return repealed


def iter_article_starts(lines: list[str]) -> Iterator[tuple[int, int]]:
    """Yield ``(line_index, article_number)`` for each article marker."""
    for index, line in enumerate(lines):
        match = ARTICLE_EN.match(line)
        if match:
            yield index, int(match.group(1))


def strip_article_marker(line: str) -> str:
    """Remove a leading 'Article N' marker, keeping any text that follows it."""
    return ARTICLE_EN.sub("", line, count=1).strip()


def rebuild_arabic_block(words: Iterable[dict], tolerance: float = 3.0) -> list[str]:
    """Reconstruct the right column as logical-order Arabic lines."""
    return [
        text
        for line in group_lines(words, tolerance)
        if (text := rebuild_rtl_line(line))
    ]
