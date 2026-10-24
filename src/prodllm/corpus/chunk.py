"""Chunking the corpus for embedding.

By article, never by a fixed token window. A window that straddles two articles
yields a chunk that cannot be cited to a single article, which defeats the
purpose of the corpus. Long articles are split on their own numbered
sub-clauses -- ``(١)``, ``(٢)`` -- and every piece keeps its article number and
citation.

``chunk_size`` and ``overlap`` are the two knobs the Module 2 sweep varies.
Both act strictly *inside* one article: overlap repeats the tail of the
previous piece at the head of the next, which recovers the context a split
destroys without ever letting a chunk span two articles. That invariant is the
whole reason a citation can be trusted, so it is not a tunable.
"""

import re
from dataclasses import dataclass
from itertools import pairwise

from prodllm.corpus.schema import Article, Chunk

# The Code's own sub-clause markers, in either digit system, with the PDF's
# occasional mismatched bracket tolerated.
_SUBCLAUSE = re.compile(r"(?=[\(\)]\s*[٠-٩0-9]{1,2}\s*[\(\)])")

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 0


@dataclass(frozen=True)
class ChunkConfig:
    """One point in the chunking search space.

    Frozen so a config can be used as a dict key and logged to MLflow as the
    identity of a run. ``embedding_model`` rides along because changing it
    invalidates the index just as surely as changing ``chunk_size`` does --
    the three are one configuration, not three independent settings.
    """

    name: str
    chunk_size: int = DEFAULT_MAX_CHARS
    overlap: int = DEFAULT_OVERLAP
    embedding_model: str = "intfloat/multilingual-e5-small"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must not be negative, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            # Otherwise each window re-emits at least as much as it consumes and
            # the packing loop never advances.
            raise ValueError(
                f"overlap ({self.overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )

    def as_params(self) -> dict[str, object]:
        """The parameter dict logged against an MLflow run."""
        return {
            "config_name": self.name,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": self.embedding_model,
        }


def split_subclauses(text: str) -> list[str]:
    """Split Arabic article text on its numbered sub-clauses."""
    parts = [part.strip() for part in _SUBCLAUSE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters of ``text``, widened to a word boundary.

    Never a mid-word fragment: a cut inside a word would put something like
    ``قادم`` at the head of the next chunk, which embeds as a different token
    than the word it came from and so makes the overlap actively harmful.
    """
    if overlap <= 0 or not text:
        return ""
    cut = max(len(text) - overlap, 0)
    if cut > 0 and not text[cut - 1].isspace():
        # The cut landed inside a word. Widen to the start of that word rather
        # than trimming to the next space: trimming can discard the tail
        # entirely when the overlap is shorter than one word, and a few
        # characters over the requested width costs nothing.
        space = text.rfind(" ", 0, cut)
        cut = space + 1 if space != -1 else 0
    return text[cut:].strip()


def split_by_length(
    text: str, max_chars: int, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Fall back to sentence-boundary packing for a long unnumbered article.

    With ``overlap`` above zero each piece after the first opens with the tail
    of the one before it, so a provision split across the boundary is still
    retrievable from both sides.
    """
    sentences = re.split(r"(?<=[.؟!।])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            carried = _tail(chunks[-1], overlap)
            current = f"{carried} {sentence}".strip() if carried else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text.strip()]


def chunk_article(
    article: Article,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn one article into one or more citable chunks.

    Repealed articles produce no chunks: there is nothing to retrieve, and
    embedding an empty record would pollute the index with a meaningless
    vector that can still be returned as a nearest neighbour.
    """
    if article.is_repealed or not article.text_ar.strip():
        return []

    pieces = (
        [article.text_ar.strip()]
        if len(article.text_ar) <= max_chars
        else split_subclauses(article.text_ar)
    )
    # Sub-clause splitting can still leave an over-long piece.
    if any(len(piece) > max_chars for piece in pieces):
        expanded: list[str] = []
        for piece in pieces:
            expanded.extend(
                split_by_length(piece, max_chars, overlap)
                if len(piece) > max_chars
                else [piece]
            )
        pieces = expanded
    # Sub-clauses are split on the Code's own markers rather than by length, so
    # they get their overlap here instead of inside split_by_length.
    elif overlap and len(pieces) > 1:
        pieces = [
            pieces[0],
            *(
                f"{_tail(previous, overlap)} {piece}".strip()
                for previous, piece in pairwise(pieces)
            ),
        ]

    total = len(pieces)
    return [
        Chunk(
            chunk_id=f"art-{article.article_number}-{index}",
            article_number=article.article_number,
            part=index,
            parts_total=total,
            text_ar=piece,
            # The English text is not split in step with the Arabic, so it is
            # attached only to the first piece rather than mis-aligned.
            text_en=article.text_en if index == 1 else "",
            citation=(
                article.citation
                if total == 1
                else f"{article.citation} ({index}/{total})"
            ),
            book=article.book,
            chapter=article.chapter,
            section=article.section,
            topic=article.topic,
            source_page=article.source_page,
        )
        for index, piece in enumerate(pieces, start=1)
    ]


def chunk_corpus(
    articles: list[Article],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, max_chars, overlap))
    return chunks


def chunk_corpus_with(articles: list[Article], config: ChunkConfig) -> list[Chunk]:
    """``chunk_corpus`` driven by a :class:`ChunkConfig`.

    The sweep calls this so that the object it logs to MLflow is the same
    object that produced the chunks, rather than two parallel sets of numbers
    that can drift apart.
    """
    return chunk_corpus(articles, config.chunk_size, config.overlap)
