"""The ASGI application.

The model is loaded **once**, here in the lifespan handler, and never inside a
request handler -- that is the single most expensive mistake available in this
module, worth a ~100x latency penalty per request.

A missing artifact is logged and the service continues in a degraded state
rather than raising. Raising would make the app unimportable without the 448 MB
model, which would make the test suite unrunnable in a fresh clone; and in
production a container that answers ``/health`` with ``model_loaded: false`` is
far more diagnosable than one that crash-loops before it can log why.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

import prodllm
from prodllm.api.middleware import CorrelationIdMiddleware
from prodllm.api.routes import router
from prodllm.config import get_settings
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.logging_conf import configure_logging
from prodllm.rag import RagEngine, VectorIndex, build_generator

logger = logging.getLogger("prodllm.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(
        level=logging.getLevelNamesMapping().get(
            settings.log_level.upper(), logging.INFO
        )
    )

    app.state.settings = settings
    app.state.embedder = None
    app.state.engine = None
    app.state.started_at = time.monotonic()

    if settings.eager_load:
        started = time.perf_counter()
        try:
            app.state.embedder = TorchEmbedder.from_settings(settings)
            logger.info(
                "Model loaded",
                extra={
                    "load_ms": round((time.perf_counter() - started) * 1000, 2),
                    "weights_dir": str(settings.weights_dir),
                    "serialization_format": (
                        app.state.embedder.info.serialization_format
                    ),
                },
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error(
                "Model unavailable; inference endpoints will return 503",
                extra={"error": str(exc), "artifacts_dir": str(settings.artifacts_dir)},
            )

        if app.state.embedder is not None:
            try:
                index = VectorIndex.load(settings.index_dir)
                app.state.engine = RagEngine(
                    embedder=app.state.embedder,
                    index=index,
                    generator=build_generator(settings),
                    top_k=settings.top_k,
                    min_score=settings.min_score,
                )
                logger.info(
                    "Corpus index loaded",
                    extra={
                        "documents_indexed": len(index),
                        "index_dir": str(settings.index_dir),
                    },
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                logger.error(
                    "Corpus index unavailable; /ask will return 503",
                    extra={"error": str(exc), "index_dir": str(settings.index_dir)},
                )

    yield

    app.state.engine = None
    app.state.embedder = None


app = FastAPI(
    title="prodllm embedding service",
    description=(
        "Arabic sentence embeddings from a quantized ONNX graph. "
        "The retrieval component of an Arabic legal document Q&A system."
    ),
    version=prodllm.__version__,
    lifespan=lifespan,
)
app.add_middleware(CorrelationIdMiddleware)
app.include_router(router)
