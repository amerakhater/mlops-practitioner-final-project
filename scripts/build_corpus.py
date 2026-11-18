"""Build the structured corpus from the Egyptian Civil Code PDF.

    uv run --group corpus python scripts/build_corpus.py \
        --pdf data/Law.pdf --out data/processed/articles.json

The PDF is raw input, not the corpus. This script turns it into one JSON record
per article so that retrieval can cite an article number -- a legal answer a
lawyer cannot verify is worthless.

What it has to work around, all confirmed against this specific file:

* **Two interleaving columns.** English left, Arabic right. A naive text dump
  mixes the two mid-sentence. The split is positional (by word midpoint), not
  script-based, because article numbers and statute references appear as digits
  inside the Arabic column.
* **Visual-order Arabic.** Glyphs come out reversed: ``مادة`` as ``ةدام``.
  Reversal is per word and only for words containing Arabic letters, so that
  ``٢٣٨`` does not become ``٨٣٢``. See ``prodllm.corpus.bidi``.
* **Repealed articles.** 55 numbers are absent because the Code says they were
  repealed (54-80 by Presidential Decree-Law 348/1952, and 389-417). Those get
  records with ``is_repealed: true`` rather than being silently missing.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from prodllm.corpus.chunk import DEFAULT_MAX_CHARS, chunk_corpus
from prodllm.corpus.parse import (
    classify_heading,
    find_repealed_ranges,
    group_lines,
    is_heading_title,
    iter_article_starts,
    ltr_line_text,
    match_arabic_article_marker,
    rebuild_arabic_block,
    split_columns,
    strip_article_marker,
)
from prodllm.corpus.schema import Article
from prodllm.corpus.validate import validate_corpus


def read_pages(pdf_path: Path, gutter: float | None) -> list[dict]:
    """Extract per-page English lines, Arabic lines and the naive text dump."""
    import pdfplumber

    pages: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            split_at = gutter if gutter is not None else page.width / 2
            left, right = split_columns(words, split_at)
            pages.append(
                {
                    "page": number,
                    "en_lines": [
                        text
                        for line in group_lines(left)
                        if (text := ltr_line_text(line))
                    ],
                    "ar_lines": rebuild_arabic_block(right),
                    "raw_text": page.extract_text() or "",
                }
            )
    return pages


def collect_english(pages: list[dict]) -> tuple[dict[int, dict], dict[int, str]]:
    """Walk the English column, segmenting it into articles and tracking headings.

    Returns the per-article English payload and the heading context in force at
    each article.
    """
    flat: list[tuple[int, str]] = [
        (page["page"], line) for page in pages for line in page["en_lines"]
    ]
    lines = [line for _, line in flat]
    starts = dict(iter_article_starts(lines))

    context: dict[str, str | None] = {
        "book": None,
        "chapter": None,
        "section": None,
        "topic": None,
    }
    articles: dict[int, dict] = {}
    current: int | None = None

    pending_title_for: str | None = None

    for index, (page_number, line) in enumerate(flat):
        heading = classify_heading(line)
        if heading is not None:
            level, text = heading
            context[level] = text
            # A new book resets what sits beneath it, so stale chapter names
            # do not leak across structural boundaries.
            if level == "book":
                context["chapter"] = context["section"] = context["topic"] = None
            elif level == "chapter":
                context["section"] = context["topic"] = None
            # "BOOK I" alone is not informative; the descriptive title is
            # printed on the line below it. Remember to pick it up next.
            pending_title_for = level
            continue

        if pending_title_for is not None:
            is_title = index not in starts and is_heading_title(line)
            if is_title:
                label = context[pending_title_for]
                context[pending_title_for] = f"{label} - {line.strip()}"
            pending_title_for = None
            if is_title:
                # The title line is metadata, not article body text.
                continue

        if index in starts:
            current = starts[index]
            articles[current] = {
                "page": page_number,
                "lines": [remainder]
                if (remainder := strip_article_marker(line))
                else [],
                "context": dict(context),
            }
            continue

        if current is not None and current in articles:
            articles[current]["lines"].append(line)

    return articles, {}


def collect_arabic(pages: list[dict]) -> dict[int, list[str]]:
    """Segment the Arabic column into articles keyed by article number.

    Heading detection is anchored and length-limited by
    ``match_arabic_article_marker``, so a body-text citation like
    "أحكام المادة ٩٨٢" is not mistaken for the start of article 982.
    """
    blocks: dict[int, list[str]] = {}
    current: int | None = None
    for page in pages:
        for line in page["ar_lines"]:
            number = match_arabic_article_marker(line)
            if number is not None:
                current = number
                blocks.setdefault(current, [])
                continue
            if current is not None:
                blocks.setdefault(current, []).append(line)
    return blocks


def build(pdf_path: Path, gutter: float | None) -> tuple[list[Article], dict]:
    pages = read_pages(pdf_path, gutter)
    repealed = find_repealed_ranges(page["raw_text"] for page in pages)
    english, _ = collect_english(pages)
    arabic = collect_arabic(pages)

    numbers = sorted(set(english) | set(arabic) | repealed)
    articles: list[Article] = []
    for number in numbers:
        payload = english.get(number, {})
        articles.append(
            Article(
                article_number=number,
                book=payload.get("context", {}).get("book"),
                chapter=payload.get("context", {}).get("chapter"),
                section=payload.get("context", {}).get("section"),
                topic=payload.get("context", {}).get("topic"),
                text_ar=" ".join(arabic.get(number, [])).strip(),
                text_en=" ".join(payload.get("lines", [])).strip(),
                is_repealed=number in repealed,
                source_page=payload.get("page", 1),
            )
        )

    stats = {
        "pages": len(pages),
        "articles": len(articles),
        "with_both_languages": sum(
            bool(a.text_ar and a.text_en) for a in articles if not a.is_repealed
        ),
    }
    return articles, stats


def print_sample(articles: list[Article], count: int, seed: int) -> None:
    """Print random articles for manual inspection.

    The handbook asks for at least 20 to be eyeballed; a fixed seed makes the
    sample reproducible so a reviewer sees the same records.
    """
    population = [a for a in articles if not a.is_repealed]
    sample = random.Random(seed).sample(population, min(count, len(population)))
    for article in sorted(sample, key=lambda a: a.article_number):
        print(f"\n--- {article.citation} (page {article.source_page}) ---")
        print(
            f"  book/chapter/section: {article.book} | {article.chapter} "
            f"| {article.section}"
        )
        print(f"  ar  : {article.text_ar[:160]}")
        print(f"  norm: {article.text_ar_normalized[:160]}")
        print(f"  en  : {article.text_en[:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/Law.pdf"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/articles.json")
    )
    parser.add_argument(
        "--chunks-out", type=Path, default=Path("data/processed/chunks.json")
    )
    parser.add_argument(
        "--gutter",
        type=float,
        default=None,
        help="Column split x-coordinate. Defaults to half the page width.",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-articles", type=int, default=1000)
    parser.add_argument(
        "--sample", type=int, default=0, help="Print N random articles to inspect."
    )
    parser.add_argument("--sample-seed", type=int, default=20260101)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write the corpus even if validation fails (for debugging only).",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        print(f"ERROR source PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    articles, stats = build(args.pdf, args.gutter)
    report = validate_corpus(articles)
    chunks = chunk_corpus(articles, args.max_chunk_chars)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            [a.model_dump(by_alias=True) for a in articles],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.chunks_out.write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {args.chunks_out}  ({args.chunks_out.stat().st_size / 1024:.0f} KB)")
    print(f"  pages scanned        : {stats['pages']}")
    print(f"  articles             : {report.total}")
    print(f"  number range         : {report.first_number}..{report.last_number}")
    print(f"  marked repealed      : {report.repealed}")
    print(f"  bilingual (non-rep.) : {stats['with_both_languages']}")
    print(f"  chunks for embedding : {len(chunks)}")
    print(f"  multi-part articles  : {sum(1 for c in chunks if c.parts_total > 1)}")
    print(
        f"  missing arabic       : {len(report.missing_arabic)} "
        f"{report.missing_arabic[:10]}"
    )
    print(
        f"  missing english      : {len(report.missing_english)} "
        f"{report.missing_english[:10]}"
    )
    print(
        f"  unexplained gaps     : {len(report.unexplained_gaps)} "
        f"{report.unexplained_gaps[:10]}"
    )

    if args.sample:
        print(
            f"\n=== {args.sample} random articles for inspection "
            f"(seed {args.sample_seed}) ==="
        )
        print_sample(articles, args.sample, args.sample_seed)

    if stats["articles"] < args.min_articles:
        print(
            f"ERROR only {report.total} articles, expected at least "
            f"{args.min_articles}; extraction likely regressed",
            file=sys.stderr,
        )
        return 1

    if not report.ok:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for problem in report.errors:
            print(f"  - {problem}", file=sys.stderr)
        if not args.allow_invalid:
            return 1
        print("  (continuing because --allow-invalid was passed)", file=sys.stderr)
    else:
        print("\nvalidation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
