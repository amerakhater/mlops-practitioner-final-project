"""Sentence-embedding pooling, kept as pure NumPy functions.

These are separate from the embedder on purpose. They hold the part of the e5
contract most likely to be got wrong -- mean-over-tokens honouring the
attention mask, *not* the CLS token, then L2 normalization -- and keeping them
free of any onnxruntime import means they are unit-testable without the 448 MB
model present.
"""

import numpy as np

# Guards against 0/0 on an all-padding row, which would otherwise yield nan and
# poison every downstream cosine similarity.
_MASK_FLOOR = 1e-9
_NORM_FLOOR = 1e-12


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average token vectors over the real (unmasked) positions.

    Args:
        last_hidden: ``(batch, sequence, hidden)`` transformer output.
        attention_mask: ``(batch, sequence)``, 1 for real tokens, 0 for padding.

    Returns:
        ``(batch, hidden)`` pooled vectors.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), _MASK_FLOOR, None)
    return summed / counts


def l2_normalize(vectors: np.ndarray, eps: float = _NORM_FLOOR) -> np.ndarray:
    """Scale each row to unit length, leaving a zero row finite."""
    norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), eps, None)
    return vectors / norms
