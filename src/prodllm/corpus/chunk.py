"""Chunking the corpus for embedding.

By article, never by a fixed token window. A window that straddles two articles
yields a chunk that cannot be cited to a single article, which defeats the
purpose of the corpus. Long articles are split on their own numbered
sub-clauses -- ``(١)``, ``(٢)`` -- and every piece keeps its article number and
citation.
"""

import re

from prodllm.corpus.schema import Article, Chunk

# The Code's own sub-clause markers, in either digit system, with the PDF's
# occasional mismatched bracket tolerated.
_SUBCLAUSE = re.compile(r"(?=[\(\)]\s*[٠-٩0-9]{1,2}\s*[\(\)])")

DEFAULT_MAX_CHARS = 1500


def split_subclauses(text: str) -> list[str]:
    """Split Arabic article text on its numbered sub-clauses."""
    parts = [part.strip() for part in _SUBCLAUSE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def split_by_length(text: str, max_chars: int) -> list[str]:
    """Fall back to sentence-boundary packing for a long unnumbered article."""
    sentences = re.split(r"(?<=[.؟!।])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text.strip()]


def chunk_article(article: Article, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
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
                split_by_length(piece, max_chars) if len(piece) > max_chars else [piece]
            )
        pieces = expanded

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
    articles: list[Article], max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, max_chars))
    return chunks
