"""Download the encoder weights into the artifacts directory.

    uv run python scripts/fetch_weights.py

Run during the image build (the ``weights`` stage of docker/Dockerfile) so the
runtime image carries its own weights. ``TorchEmbedder.from_settings`` then
loads them with ``local_files_only=True`` and the container needs no network at
startup -- which is what makes the README's three commands work on a machine
that has never seen this model, and what makes an air-gapped deployment
possible.

Two things are written:

``<artifacts>/model/``       config.json, the tokenizer files and
                             model.safetensors, in the layout
                             ``AutoModel.from_pretrained`` expects.
``<artifacts>/metadata.json``  what the service reports from ``/metadata``,
                             including the SHA-256 of the weight file. Without
                             it the endpoint reports an empty digest and the
                             image's provenance is unverifiable.

The revision is pinned. A floating ``main`` would mean two builds of the same
commit could ship different weights, which silently invalidates any retrieval
measurement taken against the earlier one.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from prodllm.config import get_settings
from prodllm.embedding.torch_embedder import sha256_of

# intfloat/multilingual-e5-small, verified to be the current HEAD of the repo.
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
WEIGHT_FILENAME = "model.safetensors"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--model-id", default=settings.source_model_id)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Commit SHA to pin. Pass 'main' only for a deliberate upgrade.",
    )
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    weights_dir = args.artifacts_dir / settings.weights_dirname
    metadata_path = args.artifacts_dir / settings.metadata_file

    print(f"fetching {args.model_id} @ {args.revision[:12]} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModel.from_pretrained(args.model_id, revision=args.revision)

    weights_dir.mkdir(parents=True, exist_ok=True)
    # safe_serialization writes model.safetensors rather than pytorch_model.bin.
    # That is the format decision of Step 4 applied to the image: safetensors
    # is read without executing any code, where a .bin is a pickle and loading
    # one from an untrusted source runs whatever it contains.
    model.save_pretrained(weights_dir, safe_serialization=True)
    tokenizer.save_pretrained(weights_dir)
    print(f"wrote {weights_dir}")

    weight_file = weights_dir / WEIGHT_FILENAME
    if not weight_file.is_file():
        print(f"ERROR expected {weight_file} after save_pretrained", file=sys.stderr)
        return 1

    # model_max_length is a very large sentinel on some tokenizers, so clamp it
    # to what the encoder's position embeddings can actually accept.
    declared = int(getattr(tokenizer, "model_max_length", settings.max_seq_length))
    max_seq_length = min(declared, int(model.config.max_position_embeddings))

    metadata = {
        "source_model_id": args.model_id,
        "revision": args.revision,
        "embedding_dim": int(model.config.hidden_size),
        "max_seq_length": max_seq_length,
        "serialization_format": "safetensors",
        "artifact_sha256": sha256_of(weight_file),
        "pooling": "mean",
        "normalized": True,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total_bytes = sum(p.stat().st_size for p in weights_dir.rglob("*") if p.is_file())
    print(f"wrote {metadata_path}")
    print(f"  weights          : {total_bytes / 1024 / 1024:.0f} MB")
    print(f"  embedding dim    : {metadata['embedding_dim']}")
    print(f"  max seq length   : {metadata['max_seq_length']}")
    print(f"  sha256           : {metadata['artifact_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
