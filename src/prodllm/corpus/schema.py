"""Corpus records.

One record per article. A RAG system built on unstructured page text cannot
cite an article number, and a legal answer that cannot be verified is
worthless, so the citation is part of the schema rather than reconstructed
downstream.

Field naming note: the handbook's field list calls the Arabic field
``text_ar``, while its illustrative JSON snippet writes ``ar_text``. ``text_ar``
is used here because it matches the field list and is symmetric with
``text_en``; ``ar_text`` is accepted as an input alias so either spelling
deserialises.
"""

from pydantic import BaseModel, ConfigDict, Field

from prodllm.corpus.normalize import normalize_arabic

CITATION_PREFIX = "Egyptian Civil Code, Article"

# A single article running past this many characters almost certainly means two
# or more articles were merged by a failed split.
IMPLAUSIBLE_LENGTH = 12_000


class Article(BaseModel):
    """A single article of the Egyptian Civil Code, bilingual."""

    model_config = ConfigDict(populate_by_name=True)

    article_number: int = Field(ge=1, description="Sortable integer, not Arabic-Indic.")
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    text_ar: str = Field(
        default="",
        validation_alias="ar_text",
        serialization_alias="text_ar",
        description="Arabic text in logical order, exactly as printed.",
    )
    text_ar_normalized: str = Field(
        default="",
        description="Retrieval copy: diacritics stripped, hamza/alef unified.",
    )
    text_en: str = ""
    is_repealed: bool = False
    source_page: int = Field(ge=1, description="1-based page in the source PDF.")
    citation: str = ""

    def model_post_init(self, _context: object) -> None:
        if not self.citation:
            self.citation = f"{CITATION_PREFIX} {self.article_number}"
        if self.text_ar and not self.text_ar_normalized:
            self.text_ar_normalized = normalize_arabic(self.text_ar)

    @property
    def is_empty(self) -> bool:
        """True when neither language has text. Legitimate only if repealed."""
        return not self.text_ar.strip() and not self.text_en.strip()

    @property
    def is_implausibly_long(self) -> bool:
        return max(len(self.text_ar), len(self.text_en)) > IMPLAUSIBLE_LENGTH


class Chunk(BaseModel):
    """An embeddable unit of text that always carries its article citation.

    Chunking is by article, not by a fixed token window: a window that straddles
    two articles produces a chunk that cannot be cited, which is the failure the
    whole corpus design exists to avoid. Long articles are split by paragraph,
    and every resulting piece keeps its article number.
    """

    chunk_id: str
    article_number: int
    part: int = Field(default=1, ge=1, description="1-based paragraph index.")
    parts_total: int = Field(default=1, ge=1)
    text_ar: str
    text_ar_normalized: str = ""
    text_en: str = ""
    citation: str
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    source_page: int = Field(ge=1)

    def model_post_init(self, _context: object) -> None:
        if self.text_ar and not self.text_ar_normalized:
            self.text_ar_normalized = normalize_arabic(self.text_ar)
