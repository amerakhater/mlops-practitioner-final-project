"""Add new documents to the index without rebuilding it.

    uv run python scripts/reindex.py --chunks tests/fixtures/new_articles.sample.json
    uv run python scripts/reindex.py --chunks data/processed/chunks.json --dry-run

The offline half of batch re-indexing. ``scripts/build_index.py`` embeds the
whole corpus and writes a new index; this embeds only the documents the index
does not already hold and appends them.

Why that distinction is worth a second script: re-embedding ~1100 chunks to
publish one new article takes minutes, and it rewrites vectors nothing asked to
change. An append touches only the new rows, so publishing an amendment costs
roughly the time to embed the amendment.

The job is **idempotent**: documents whose chunk id is already indexed are
reported as skipped and not embedded again. So pointing this at the full corpus
file is a safe no-op, and re-running it after a partial failure cannot create
duplicates -- which matters more than disk space, because ``retrieve`` dedupes
by article number over a fixed candidate pool, and a doubled chunk silently
consumes a slot that a genuine article should have had.

Writing the artifact is the default here, unlike the live ``/reindex``
endpoint: this script exists to produce a new index on disk. ``--dry-run``
reports what would be added and writes nothing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from prodllm.config import get_settings
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.rag import RagEngine, VectorIndex


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="JSON file holding a list of chunk objects to add.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.index_dir,
        help="The index to extend. Defaults to the configured index dir.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be added and write nothing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON, for a CI step to assert on.",
    )
    args = parser.parse_args()

    if not args.chunks.is_file():
        print(f"ERROR chunks file not found: {args.chunks}", file=sys.stderr)
        return 2

    payload = json.loads(args.chunks.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        print(
            f"ERROR {args.chunks} must hold a non-empty JSON list of chunks",
            file=sys.stderr,
        )
        return 2

    try:
        index = VectorIndex.load(args.index)
    except FileNotFoundError as exc:
        print(
            f"ERROR {exc}\nRun scripts/build_index.py first -- this script "
            "extends an existing index, it does not create one.",
            file=sys.stderr,
        )
        return 2

    print(f"index    : {args.index}")
    print(f"  documents indexed : {index.document_count}")
    print(f"  vectors           : {len(index)}")
    print(f"candidates in {args.chunks.name}: {len(payload)}")

    known = index.chunk_ids
    fresh = [
        chunk
        for chunk in payload
        if (chunk.get("chunk_id") or chunk.get("article_number")) not in known
    ]
    print(f"  already indexed   : {len(payload) - len(fresh)}")
    print(f"  new               : {len(fresh)}")

    if not fresh:
        print("nothing to do")
        if args.json:
            print(
                json.dumps(
                    {
                        "added": 0,
                        "skipped": len(payload),
                        "vectors_added": 0,
                        "documents_indexed": index.document_count,
                    }
                )
            )
        return 0

    if args.dry_run:
        print("dry run: no model loaded, nothing written")
        for chunk in fresh[:10]:
            print(
                f"  would add {chunk.get('chunk_id')} (article "
                f"{chunk.get('article_number')})"
            )
        if len(fresh) > 10:
            print(f"  ... and {len(fresh) - 10} more")
        return 0

    print(f"loading model ({settings.source_model_id}) ...")
    started = time.perf_counter()
    embedder = TorchEmbedder.from_settings(settings)
    print(f"  model ready in {time.perf_counter() - started:.1f}s")

    # Dimension is checked inside VectorIndex: new documents embedded with a
    # different encoder cannot be appended to an index built with another one.
    engine = RagEngine(embedder=embedder, index=index)

    print(f"embedding {len(fresh)} new documents ...")
    started = time.perf_counter()
    result = engine.add_documents(payload, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    print(f"  embedded in {elapsed:.2f}s")

    index.save(args.index)
    print(f"wrote {args.index}")
    print(f"  added             : {result.added}")
    print(f"  skipped           : {result.skipped}")
    print(f"  vectors added     : {result.vectors_added}")
    print(f"  documents indexed : {result.documents_indexed}")

    if args.json:
        print(json.dumps(result.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
