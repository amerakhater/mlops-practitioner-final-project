"""A transformers cross-encoder, used for both the teacher and the student.

One class, not two. The teacher (``BAAI/bge-reranker-v2-m3``) and the student
distilled from it differ in their weights and their width, not in how they are
called: both are sequence-classification heads over a single logit, and both
tokenize the query and the passage as a *pair* so that attention runs across
the boundary. Giving them separate classes would duplicate the tokenize ->
forward -> squeeze path and invite the two to drift, which in a distillation
setup is the one divergence that silently corrupts the training labels -- the
student would be learning to imitate scores the teacher never produced under
the student's own preprocessing.

As in ``prodllm.embedding.torch_embedder``, all file loading lives in a
classmethod and ``__init__`` takes a ready-made model and tokenizer, so the
scoring path is testable with mocks and no weights on disk.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from prodllm.instrumentation import log_latency
from prodllm.rerank.base import Reranker, RerankerInfo

logger = logging.getLogger("prodllm.rerank")

# The teacher. Multilingual (XLM-RoBERTa-large, 568M parameters) because this
# corpus is Arabic and English and a question in either language has to be
# scored against a passage that may be in the other. An English-only
# cross-encoder would be a stronger model on paper and useless here.
DEFAULT_TEACHER_MODEL = "BAAI/bge-reranker-v2-m3"

# Pairs per forward pass. Small because the sequences are long and this runs
# on CPU in the serving path; raising it trades latency variance for
# throughput, which is the wrong trade for a per-request re-rank of ~20 pairs.
DEFAULT_BATCH_SIZE = 8


class CrossEncoderReranker(Reranker):
    """Scores (query, passage) pairs with a sequence-classification head."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        info: RerankerInfo,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._info = info
        self._batch_size = batch_size

    @property
    def info(self) -> RerankerInfo:
        return self._info

    @log_latency("rerank.score")
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw relevance logits, one per passage, in input order."""
        if not passages:
            return []

        scores: list[float] = []
        for start in range(0, len(passages), self._batch_size):
            batch = passages[start : start + self._batch_size]
            scores.extend(self._score_batch(query, batch))
        return scores

    def _score_batch(self, query: str, passages: Sequence[str]) -> list[float]:
        # The query is passed as the *first* sequence and the passages as the
        # second, with the default `longest_first` truncation. That ordering is
        # load-bearing: it means a long article is trimmed and the question
        # survives intact. Truncating the other way round would silently score
        # a half-stated question, and the failure is invisible -- the model
        # still returns a confident number.
        encoded = self._tokenizer(
            [query] * len(passages),
            list(passages),
            padding=True,
            truncation=True,
            max_length=self._info.max_seq_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self._model(**encoded).logits
        # (batch, 1) for a regression head, (batch,) if already squeezed.
        return logits.reshape(-1).float().tolist()

    @classmethod
    def load(
        cls,
        model_id: str | Path,
        *,
        max_seq_length: int = 512,
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_files_only: bool = False,
        distilled_from: str | None = None,
    ) -> "CrossEncoderReranker":
        """Load a cross-encoder from the hub or from a local directory.

        ``local_files_only`` defaults to False here, unlike the embedder, for a
        deliberate reason: the embedder is loaded inside the serving container
        where an unexpected network call is a provenance hole, whereas this
        classmethod's first caller is the offline distillation job, which has
        to be able to fetch the teacher. The serving path passes True.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        source = str(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            source, local_files_only=local_files_only
        )
        model.eval()

        parameters = sum(p.numel() for p in model.parameters())
        logger.info(
            "Loaded cross-encoder re-ranker",
            extra={
                "model_id": source,
                "parameters": parameters,
                "distilled_from": distilled_from,
            },
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            info=RerankerInfo(
                model_id=source,
                parameters=parameters,
                max_seq_length=max_seq_length,
                distilled_from=distilled_from,
            ),
            batch_size=batch_size,
        )
