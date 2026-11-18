"""The serving backend: a transformers encoder plus our own pooling.

``transformers`` directly rather than ``sentence-transformers`` is deliberate.
The pooling contract for e5 -- mean over unmasked tokens, *not* the CLS token,
then L2 normalization -- stays in ``prodllm.pooling``, where it is unit-tested
on hand-built arrays, instead of being delegated to a library's ``modules.json``.

All file loading lives in ``from_settings``; ``__init__`` takes a ready-made
model and tokenizer. That seam is what lets the tokenize -> forward -> pool ->
normalize path be tested with mocks and no weights on disk.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from prodllm.config import Settings
from prodllm.embedding.base import DEFAULT_PREFIX, Embedder, ModelInfo
from prodllm.instrumentation import log_latency
from prodllm.pooling import l2_normalize, mean_pool

logger = logging.getLogger("prodllm.embedding")


class TorchEmbedder(Embedder):
    """Embeds text with a transformers encoder held in memory."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        info: ModelInfo,
        max_seq_length: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._info = info

    @property
    def info(self) -> ModelInfo:
        return self._info

    @log_latency("embedder.embed")
    def embed(
        self, texts: Sequence[str], *, prefix: str = DEFAULT_PREFIX
    ) -> list[list[float]]:
        if not texts:
            return []

        batch = self._tokenizer(
            [prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self._model(**batch)

        hidden = output.last_hidden_state.detach().cpu().numpy()
        mask = batch["attention_mask"].detach().cpu().numpy()
        return l2_normalize(mean_pool(hidden, mask)).tolist()

    @classmethod
    def from_settings(cls, settings: Settings) -> "TorchEmbedder":
        """Load the model from local files only.

        ``local_files_only`` is not a nicety: without it a misconfigured
        artifacts dir makes the container silently reach out to Hugging Face at
        startup, which fails in an air-gapped deployment and makes the image's
        provenance unverifiable.

        Raises:
            FileNotFoundError: naming the missing path, so a bad
                ``PRODLLM_ARTIFACTS_DIR`` is obvious from the startup log.
        """
        from transformers import AutoModel, AutoTokenizer

        model_dir = settings.weights_dir
        baked_in = model_dir.is_dir()

        if baked_in:
            source: str | Path = model_dir
            local_only = True
        elif settings.allow_download:
            # Thin image: the weights are fetched on first start and cached in
            # the Hugging Face cache, which should be a mounted volume so the
            # download survives container restarts.
            source = settings.source_model_id
            local_only = False
            logger.warning(
                "Weights not baked into the image; downloading on first start",
                extra={
                    "source_model_id": settings.source_model_id,
                    "weights_dir": str(model_dir),
                },
            )
        else:
            raise FileNotFoundError(
                f"weights directory is missing: {model_dir} "
                "(and PRODLLM_ALLOW_DOWNLOAD is false)"
            )

        metadata: dict[str, object] = {}
        if settings.metadata_path.is_file():
            metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))

        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        model = AutoModel.from_pretrained(source, local_files_only=local_only)
        model.eval()
        torch.set_num_threads(settings.torch_num_threads)

        info = ModelInfo(
            source_model_id=metadata.get("source_model_id", settings.source_model_id),
            embedding_dim=int(metadata.get("embedding_dim", model.config.hidden_size)),
            max_seq_length=int(metadata.get("max_seq_length", settings.max_seq_length)),
            serialization_format=metadata.get("serialization_format", "safetensors"),
            artifact_sha256=metadata.get("artifact_sha256", ""),
            pooling=metadata.get("pooling", "mean"),
            normalized=bool(metadata.get("normalized", True)),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            info=info,
            max_seq_length=info.max_seq_length,
        )


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading the whole thing into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
