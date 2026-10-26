"""The re-ranker contract.

A re-ranker is the second stage of retrieval. The FAISS index scores a query
against each article *independently* -- the query vector and the passage vector
were computed without ever seeing one another -- which is what makes it fast
enough to sweep 2183 vectors per request and also what makes its scores
non-discriminative on this corpus. ``prodllm.rag.DEFAULT_MIN_SCORE`` records
the measurement: on-topic questions score 0.838-0.899 and off-topic ones score
0.789-0.859, so the ranges overlap and no absolute threshold separates them.

A cross-encoder removes exactly that limitation. It concatenates the query and
the passage into one sequence and runs full attention across the pair, so the
score is computed with both sides visible. The cost is that it cannot be
precomputed: every (query, passage) pair is a forward pass, which is why it
re-ranks a candidate list rather than replacing the index.

An abstract base class rather than a Protocol, for the same reason as
``prodllm.embedding.base.Embedder``: ``isinstance`` checks work, and the three
implementations here -- teacher, distilled student and fake -- share one
documented interface.
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerInfo:
    """What ``/metadata`` publishes about the re-ranker actually loaded.

    ``parameters`` and ``distilled_from`` are here rather than left implicit
    because the whole point of the distilled student is that it is smaller than
    its teacher, and a claim like that belongs where a caller can check it
    instead of only in a report.
    """

    model_id: str
    parameters: int
    max_seq_length: int
    distilled_from: str | None = None


class Reranker(abc.ABC):
    """Scores (query, passage) pairs jointly and reorders them."""

    @property
    @abc.abstractmethod
    def info(self) -> RerankerInfo:
        """Describe the loaded re-ranker."""

    @abc.abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against ``query``, in input order.

        Higher is more relevant. The scale is the implementation's own -- these
        are raw logits, not probabilities, and they are **not** comparable
        across models. Only the ordering within one call is meaningful, which
        is why ``rerank`` returns positions rather than the caller being
        invited to threshold the numbers.
        """

    def rerank(
        self, query: str, passages: Sequence[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` best-first, truncated to ``top_k``.

        Positions rather than the passages themselves: the caller holds richer
        objects than the strings it handed in -- ``RetrievedArticle`` carries
        citations and page numbers -- and returning indices lets it reorder
        those without this module having to know what they are.

        Implemented once here so that an implementation only has to supply
        ``score``. The sort is stable, so passages the model scores equally
        keep the order the first-stage retriever put them in.
        """
        if not passages:
            return []
        scores = self.score(query, passages)
        if len(scores) != len(passages):
            raise ValueError(
                f"re-ranker returned {len(scores)} scores for {len(passages)} passages"
            )
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        limit = len(order) if top_k is None else min(top_k, len(order))
        return [(i, scores[i]) for i in order[:limit]]
