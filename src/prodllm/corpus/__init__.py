"""Corpus construction from the source Civil Code PDF.

Everything here is pure: it operates on word dictionaries and strings, never on
a PDF. The pdfplumber I/O lives in ``scripts/build_corpus.py``, which keeps
pdfplumber out of the runtime image and makes this logic unit-testable without
a 2.5 MB fixture.
"""

from prodllm.corpus.bidi import rebuild_rtl_line, reverse_visual_arabic
from prodllm.corpus.chunk import chunk_article, chunk_corpus
from prodllm.corpus.normalize import normalize_arabic
from prodllm.corpus.schema import Article, Chunk
from prodllm.corpus.validate import CorpusReport, validate_corpus

__all__ = [
    "Article",
    "Chunk",
    "CorpusReport",
    "chunk_article",
    "chunk_corpus",
    "normalize_arabic",
    "rebuild_rtl_line",
    "reverse_visual_arabic",
    "validate_corpus",
]
