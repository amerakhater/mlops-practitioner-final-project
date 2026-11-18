"""The HTTP endpoints.

``/ask`` is the contract for this track: a question in, an answer plus
verifiable article citations out.

Handlers are ``async def``, but the work inside them -- a torch forward pass
and a FAISS search -- is blocking CPU work. Running it directly on the event
loop would stall every other request, including the correlation-ID middleware,
so it is handed to a threadpool with ``run_in_threadpool``. That keeps the
handlers asynchronous without pretending CPU work is I/O.
"""

import logging
import time

from fastapi import APIRouter, Request, status
from starlette.concurrency import run_in_threadpool

import prodllm
from prodllm.api.deps import (
    EmbedderDep,
    EngineDep,
    OptionalEmbedderDep,
    OptionalEngineDep,
    SettingsDep,
)
from prodllm.api.schemas import (
    AskRequest,
    AskResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    SourceArticle,
)
from prodllm.embedding.base import SUPPORTED_PREFIXES

logger = logging.getLogger("prodllm.api")
router = APIRouter()

_SERVICE_UNAVAILABLE = {status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}}


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(
    request: Request, embedder: OptionalEmbedderDep, engine: OptionalEngineDep
) -> HealthResponse:
    """Liveness.

    Deliberately 200 even when degraded: a container that crash-loops before
    it can log why is far harder to diagnose than one that reports its state.
    Readiness is ``documents_indexed``, which the image's HEALTHCHECK asserts.
    """
    started_at = getattr(request.app.state, "started_at", None)
    uptime = 0.0 if started_at is None else time.monotonic() - started_at
    indexed = engine.documents_indexed if engine is not None else 0
    ready = engine is not None and embedder is not None and indexed > 0
    return HealthResponse(
        status="healthy" if ready else "degraded",
        documents_indexed=indexed,
        version=prodllm.__version__,
        model_loaded=embedder is not None,
        uptime_seconds=round(uptime, 3),
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses=_SERVICE_UNAVAILABLE,
    tags=["qa"],
)
async def ask(body: AskRequest, engine: EngineDep) -> AskResponse:
    """Answer a question about the Egyptian Civil Code, with citations.

    When nothing clears the relevance threshold the service says so and
    returns no sources, rather than answering from the least-irrelevant
    article it could find. For a legal assistant, declining is correct.
    """
    start = time.perf_counter()
    result = await run_in_threadpool(engine.answer, body.question)
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    logger.info(
        "Question answered",
        extra={
            "grounded": result.grounded,
            "n_sources": len(result.sources),
            "latency_ms": latency_ms,
        },
    )
    return AskResponse(
        answer=result.answer,
        sources=[SourceArticle(**source.as_source()) for source in result.sources],
        grounded=result.grounded,
        latency_ms=latency_ms,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    responses=_SERVICE_UNAVAILABLE,
    tags=["model"],
)
async def metadata(embedder: EmbedderDep, settings: SettingsDep) -> MetadataResponse:
    """Describe the artifact actually loaded, not what the config asked for."""
    info = embedder.info
    return MetadataResponse(
        service_version=prodllm.__version__,
        source_model_id=info.source_model_id,
        embedding_dim=info.embedding_dim,
        max_seq_length=info.max_seq_length,
        serialization_format=info.serialization_format,
        artifact_sha256=info.artifact_sha256,
        pooling=info.pooling,
        normalized=info.normalized,
        supported_prefixes=list(SUPPORTED_PREFIXES),
        max_batch_size=settings.max_batch_size,
        max_text_chars=settings.max_text_chars,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=_SERVICE_UNAVAILABLE,
    tags=["embeddings"],
)
async def predict(body: PredictRequest, embedder: EmbedderDep) -> PredictResponse:
    """Embed a single text. The generic rubric endpoint, kept alongside /ask."""
    start = time.perf_counter()
    vectors = await run_in_threadpool(
        lambda: embedder.embed([body.text], prefix=body.prefix)
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return PredictResponse(
        embedding=vectors[0],
        dim=len(vectors[0]),
        source_model_id=embedder.info.source_model_id,
        latency_ms=latency_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses=_SERVICE_UNAVAILABLE,
    tags=["embeddings"],
)
async def predict_batch(
    body: BatchPredictRequest, embedder: EmbedderDep
) -> BatchPredictResponse:
    start = time.perf_counter()
    vectors = await run_in_threadpool(
        lambda: embedder.embed(body.texts, prefix=body.prefix)
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return BatchPredictResponse(
        embeddings=vectors,
        dim=len(vectors[0]) if vectors else 0,
        count=len(vectors),
        source_model_id=embedder.info.source_model_id,
        latency_ms=latency_ms,
    )
