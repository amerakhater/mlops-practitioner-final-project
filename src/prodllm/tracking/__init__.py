"""MLflow-facing code: the registered chunking model.

Separate from ``prodllm.eval`` and, like it, never imported by the service.
``prodllm.tracking`` needs mlflow installed; ``prodllm.api`` must not, or the
runtime image grows a tracking client it never calls.
"""

from prodllm.tracking.chunking_model import ChunkingModel, log_chunking_model

__all__ = ["ChunkingModel", "log_chunking_model"]
