"""The embedder contract.

An abstract base class rather than a Protocol, so that ``isinstance`` checks
work and the two implementations share one documented interface.
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

# e5 models are trained with these instruction prefixes and degrade measurably
# without them. The trailing space is part of the contract, not a typo.
SUPPORTED_PREFIXES: Final[tuple[str, ...]] = ("query: ", "passage: ")
DEFAULT_PREFIX: Final[str] = "query: "


@dataclass(frozen=True)
class ModelInfo:
    """What ``/metadata`` publishes about the artifact actually loaded.

    ``serialization_format`` records how the weights on disk were stored --
    ``safetensors`` or ``pickle``. It is the outcome of the module's format
    decision, so it is published rather than left implicit: a consumer can tell
    from the API whether the service loaded a format that can execute code.
    """

    source_model_id: str
    embedding_dim: int
    max_seq_length: int
    serialization_format: str
    artifact_sha256: str
    pooling: str = "mean"
    normalized: bool = True


class Embedder(abc.ABC):
    """Turns text into normalized sentence vectors."""

    @property
    @abc.abstractmethod
    def info(self) -> ModelInfo:
        """Describe the loaded model."""

    @abc.abstractmethod
    def embed(
        self, texts: Sequence[str], *, prefix: str = DEFAULT_PREFIX
    ) -> list[list[float]]:
        """Embed ``texts``, returning one unit-norm vector per input, in order."""
