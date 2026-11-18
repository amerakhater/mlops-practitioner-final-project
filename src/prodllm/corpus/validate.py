"""Validate the corpus before anything is embedded.

A malformed corpus does not announce itself. It surfaces three modules later as
retrieval quality nobody can explain, by which point the embeddings, the vector
store and the evaluation set have all been built on it. So the checks run as a
gate, and an unexplained gap is an error rather than a warning.
"""

from dataclasses import dataclass, field

from prodllm.corpus.schema import Article


@dataclass
class CorpusReport:
    """The outcome of validating a corpus."""

    total: int = 0
    repealed: int = 0
    first_number: int | None = None
    last_number: int | None = None
    duplicates: list[int] = field(default_factory=list)
    unexplained_gaps: list[int] = field(default_factory=list)
    explained_gaps: list[int] = field(default_factory=list)
    missing_arabic: list[int] = field(default_factory=list)
    missing_english: list[int] = field(default_factory=list)
    implausibly_long: list[int] = field(default_factory=list)
    known_defects: list[int] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        """Conditions that must block embedding."""
        problems: list[str] = []
        if self.duplicates:
            problems.append(f"duplicate article numbers: {self.duplicates[:10]}")
        if self.unexplained_gaps:
            problems.append(
                f"{len(self.unexplained_gaps)} unexplained gaps: "
                f"{self.unexplained_gaps[:10]}"
            )
        if self.missing_arabic:
            problems.append(
                f"{len(self.missing_arabic)} records with no Arabic text: "
                f"{self.missing_arabic[:10]}"
            )
        if self.implausibly_long:
            problems.append(
                f"{len(self.implausibly_long)} implausibly long records "
                f"(likely a failed split): {self.implausibly_long[:10]}"
            )
        return problems

    @property
    def ok(self) -> bool:
        return not self.errors


# Defects in the source PDF itself, not in the parser. Listed explicitly so
# they stay visible in review rather than being absorbed by a lowered
# threshold, and so a NEW defect still fails the build.
#
# 1022: printed "Article1022" with no space, and its Arabic heading is missing
#       from the right column altogether (verified on page 147).
#  452: the English column has no text for this article (verified on page 59).
KNOWN_MISSING_ARABIC: frozenset[int] = frozenset({1022})
KNOWN_MISSING_ENGLISH: frozenset[int] = frozenset({452})


def validate_corpus(
    articles: list[Article],
    known_missing_arabic: frozenset[int] = KNOWN_MISSING_ARABIC,
) -> CorpusReport:
    """Check structural invariants over the whole corpus.

    A gap in the numbering is "explained" when the document declared those
    articles repealed, which is why repealed articles are flagged and kept
    rather than dropped -- a dropped article is indistinguishable from a
    parsing failure.
    """
    report = CorpusReport(total=len(articles))
    if not articles:
        return report

    by_number: dict[int, Article] = {}
    for article in articles:
        if article.article_number in by_number:
            report.duplicates.append(article.article_number)
        by_number[article.article_number] = article

    numbers = sorted(by_number)
    report.first_number, report.last_number = numbers[0], numbers[-1]
    report.repealed = sum(1 for a in articles if a.is_repealed)

    present = set(numbers)
    for number in range(numbers[0], numbers[-1] + 1):
        if number in present:
            continue
        # Absent entirely: only acceptable if a neighbour marks it repealed.
        report.unexplained_gaps.append(number)

    for article in articles:
        if article.is_repealed:
            report.explained_gaps.append(article.article_number)
            continue
        if not article.text_ar.strip():
            if article.article_number in known_missing_arabic:
                report.known_defects.append(article.article_number)
            else:
                report.missing_arabic.append(article.article_number)
        if not article.text_en.strip():
            report.missing_english.append(article.article_number)
        if article.is_implausibly_long:
            report.implausibly_long.append(article.article_number)

    return report
