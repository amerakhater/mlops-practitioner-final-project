"""A deterministic, artifact-free re-ranker.

In ``src/`` rather than ``tests/`` for the same reason as ``FakeEmbedder``: the
API and engine suites need to exercise the two-stage retrieval path without a
2.2 GB cross-encoder on disk, and the benchmark harness uses it to measure what
the re-ranking *plumbing* costs with the forward pass subtracted out.

The scores are meaningless as relevance but are stable across processes, which
is what lets a test assert that re-ranking reordered the candidates rather than
merely that it ran.
"""

import hashlib
from collections.abc import Sequence

from prodllm.rerank.base import Reranker, RerankerInfo


class FakeReranker(Reranker):
    """Hash-seeded scores, same interface as the real thing."""

    def __init__(self, model_id: str = "fake/deterministic") -> None:
        self._info = RerankerInfo(
            model_id=model_id,
            parameters=0,
            max_seq_length=512,
            distilled_from=None,
        )

    @property
    def info(self) -> RerankerInfo:
        return self._info

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for passage in passages:
            digest = hashlib.sha256(f"{query}\x00{passage}".encode()).digest()
            # Map the first four bytes into [0, 1). Deterministic across runs
            # and across processes, which hash() is not -- PYTHONHASHSEED
            # randomization would make an ordering assertion flaky.
            scores.append(int.from_bytes(digest[:4], "big") / 2**32)
        return scores
