"""Embed the article chunks and write the FAISS index.

    uv run python scripts/build_index.py

Reads data/processed/chunks.json (produced by build_corpus.py), embeds every
chunk with the configured model, and writes artifacts/index/. The service
loads that index at startup; it never embeds the corpus on boot.

Embedding uses the "passage: " prefix and the normalized Arabic text, matching
what RagEngine.retrieve does with "query: " at query time. Mismatching those
two is a silent quality bug: retrieval still returns results, just worse ones.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from prodllm.config import get_settings
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.rag import RagEngine


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=settings.corpus_chunks_path)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Embed only the first N chunks (smoke test).",
    )
    args = parser.parse_args()

    if not args.chunks.is_file():
        print(
            f"ERROR chunks file not found: {args.chunks}\n"
            "Run scripts/build_corpus.py first.",
            file=sys.stderr,
        )
        return 2

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        print("ERROR no chunks to embed", file=sys.stderr)
        return 1

    print(f"loading model ({settings.source_model_id}) ...")
    started = time.perf_counter()
    embedder = TorchEmbedder.from_settings(settings)
    print(f"  model ready in {time.perf_counter() - started:.1f}s")

    print(f"embedding {len(chunks)} chunks (batch size {args.batch_size}) ...")
    started = time.perf_counter()
    index = RagEngine.build_index(chunks, embedder, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    print(f"  embedded in {elapsed:.1f}s ({len(chunks) / elapsed:.1f} chunks/s)")

    index.save(args.out)
    total_bytes = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    print(f"wrote {args.out}  ({total_bytes / 1024:.0f} KB)")
    print(f"  documents indexed : {index.document_count}")
    # One vector per language per chunk, so this exceeds the document count.
    print(f"  vectors           : {len(index)}")
    print(f"  embedding dim     : {embedder.info.embedding_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
