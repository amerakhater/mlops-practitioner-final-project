"""Embedding backends: the abstract contract, the ONNX implementation, and a fake."""

from prodllm.embedding.base import SUPPORTED_PREFIXES, Embedder, ModelInfo

__all__ = ["SUPPORTED_PREFIXES", "Embedder", "ModelInfo"]
