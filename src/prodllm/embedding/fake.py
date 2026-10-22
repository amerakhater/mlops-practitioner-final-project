"""A deterministic, artifact-free embedder.

It lives in ``src/`` rather than ``tests/`` for two reasons: the API test suite
uses it to exercise every route with no 448 MB model on disk, and
``scripts/bench_api.py --fake`` uses it to measure HTTP and JSON-serialization
overhead with inference subtracted out. Serializing 384 floats per request is a
real share of ``/predict`` latency and is worth being able to isolate.

The vectors are meaningless as embeddings but are stable across processes, which
is what makes order- and determinism-assertions in the tests possible.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from prodllm.embedding.base import DEFAULT_PREFIX, Embedder, ModelInfo
from prodllm.pooling import l2_normalize

_DEFAULT_DIM = 384


class FakeEmbedder(Embedder):
    """Hash-seeded random unit vectors, same interface as the real thing."""

    def __init__(
        self, dim: int = _DEFAULT_DIM, source_model_id: str = "fake/deterministic"
    ) -> None:
        self._info = ModelInfo(
            source_model_id=source_model_id,
            embedding_dim=dim,
            max_seq_length=512,
            serialization_format="none",
            artifact_sha256="0" * 64,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def embed(
        self, texts: Sequence[str], *, prefix: str = DEFAULT_PREFIX
    ) -> list[list[float]]:
        if not texts:
            return []
        dim = self._info.embedding_dim
        rows = np.empty((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            digest = hashlib.sha256((prefix + text).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rows[index] = np.random.default_rng(seed).normal(size=dim)
        return l2_normalize(rows).tolist()
