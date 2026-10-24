"""What the Bento actually does, with no ``bentoml`` import anywhere in it.

The BentoML decorators in ``service.py`` are a transport binding: they parse a
request, call something, and serialize the result. Everything worth testing --
degraded startup, the asynchronous answer path, the streaming path, a batch
re-index and what the release channel reports -- is here instead, behind a
plain class.

That split is deliberate. Exercising this layer needs no server, no port and no
BentoML runtime, so the Module 3 behaviour is covered by the same offline
``pytest`` run as everything else, and ``tests/`` stays free of the sleeps and
socket polling that make a service test suite flaky.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

import prodllm
from prodllm.config import Settings, get_settings
from prodllm.embedding.base import Embedder
from prodllm.embedding.torch_embedder import TorchEmbedder
from prodllm.rag import (
    Answer,
    RagEngine,
    ReindexResult,
    VectorIndex,
    build_generator,
)
from prodllm.streaming import StreamGenerator, build_stream_generator, stream_answer

logger = logging.getLogger("prodllm.serving")

# Re-index batch size. Smaller than the offline job's 32: a live re-index
# competes with request traffic for the same CPU, and a long embed call inside
# the threadpool is a long time during which a worker thread is unavailable.
REINDEX_BATCH_SIZE = 16


class ServiceUnavailable(RuntimeError):
    """Raised when an endpoint needs an artifact that startup could not load.

    ``service.py`` is the only place that knows this maps to HTTP 503. Keeping
    the status code out of here is what lets this layer be tested as a plain
    object.
    """


class RagServiceCore:
    """The loaded model, index and generators, plus the operations over them.

    Constructed once per worker. Loading in the constructor rather than on
    first request is the same decision ``prodllm.api.main`` documents, and for
    the same reason: a model load inside a request handler costs roughly 100x
    per-request latency.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        engine: RagEngine | None = None,
        stream_generator: StreamGenerator | None = None,
    ) -> None:
        """Build the core, loading from disk only what was not injected.

        Every collaborator is optional so a test can supply a fake embedder and
        an in-memory index and never touch the 448 MB artifact -- the same seam
        ``prodllm.api.deps`` provides for the FastAPI app.
        """
        self.settings = settings or get_settings()
        self.started_at = time.monotonic()
        self.embedder = embedder
        self.engine = engine
        self._stream_generator = stream_generator

        if self.embedder is None and self.settings.eager_load:
            self.embedder = self._load_embedder()
        if self.engine is None and self.embedder is not None:
            self.engine = self._load_engine(self.embedder)
        if self._stream_generator is None:
            self._stream_generator = build_stream_generator(self.settings)

        logger.info(
            "Serving core ready",
            extra={
                "release_channel": self.settings.release_channel,
                "ready": self.ready,
                "documents_indexed": self.documents_indexed,
                "stream_backend": type(self._stream_generator).__name__,
            },
        )

    # ------------------------------------------------------------- loading

    def _load_embedder(self) -> Embedder | None:
        """Load the encoder, or return ``None`` and stay up.

        Degrading rather than raising is the project's standing choice: a
        container reporting ``model_loaded: false`` on /health can be
        diagnosed, whereas one that crash-loops before it logs cannot.
        """
        started = time.perf_counter()
        try:
            embedder = TorchEmbedder.from_settings(self.settings)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error(
                "Model unavailable; /ask will return 503",
                extra={
                    "error": str(exc),
                    "weights_dir": str(self.settings.weights_dir),
                },
            )
            return None
        logger.info(
            "Model loaded",
            extra={
                "load_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialization_format": embedder.info.serialization_format,
            },
        )
        return embedder

    def _load_engine(self, embedder: Embedder) -> RagEngine | None:
        try:
            index = VectorIndex.load(self.settings.index_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error(
                "Corpus index unavailable; /ask will return 503",
                extra={"error": str(exc), "index_dir": str(self.settings.index_dir)},
            )
            return None
        logger.info(
            "Corpus index loaded",
            extra={
                "documents_indexed": index.document_count,
                "vectors": len(index),
            },
        )
        return RagEngine(
            embedder=embedder,
            index=index,
            generator=build_generator(self.settings),
            top_k=self.settings.top_k,
            min_score=self.settings.min_score,
        )

    # -------------------------------------------------------------- status

    @property
    def ready(self) -> bool:
        """Whether the service can actually answer a question."""
        return self.engine is not None and self.documents_indexed > 0

    @property
    def documents_indexed(self) -> int:
        return 0 if self.engine is None else self.engine.documents_indexed

    def health(self) -> dict[str, Any]:
        """Liveness, plus the release identity that makes a canary readable.

        ``release_channel`` is here rather than only in the logs so a rollout
        check can confirm which variant answered without reading the log
        stream -- during a canary that is the first question anyone asks.
        """
        return {
            "status": "healthy" if self.ready else "degraded",
            "documents_indexed": self.documents_indexed,
            "version": prodllm.__version__,
            "model_loaded": self.embedder is not None,
            "release_channel": self.settings.release_channel,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
        }

    def _require_engine(self) -> RagEngine:
        if self.engine is None:
            raise ServiceUnavailable(
                "the legal corpus index is not loaded; run "
                "scripts/build_index.py and restart, or check "
                "PRODLLM_ARTIFACTS_DIR"
            )
        return self.engine

    # ------------------------------------------------------------ answering

    async def answer(self, question: str) -> tuple[Answer, float]:
        """Answer ``question``, returning the answer and its latency in ms.

        ``run_in_threadpool`` because the work is a torch forward pass and a
        FAISS search -- blocking CPU, not I/O. Awaiting it directly on the
        event loop would stall every other in-flight request, which is exactly
        the failure the 50-user load test is there to catch.
        """
        engine = self._require_engine()
        start = time.perf_counter()
        result = await run_in_threadpool(engine.answer, question)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        logger.info(
            "Question answered",
            extra={
                "grounded": result.grounded,
                "n_sources": len(result.sources),
                "latency_ms": latency_ms,
                "release_channel": self.settings.release_channel,
            },
        )
        return result, latency_ms

    async def astream(self, question: str) -> AsyncIterator[str]:
        """Stream an answer fragment by fragment.

        Retrieval is awaited in the threadpool first, so the citations are
        settled before a single token is emitted. That ordering is not an
        implementation detail: a caller cannot verify a claim against an
        article that has not been chosen yet, and emitting prose first would
        make the stream look grounded before it is.
        """
        engine = self._require_engine()
        start = time.perf_counter()
        passages = await run_in_threadpool(engine.retrieve, question)
        retrieve_ms = round((time.perf_counter() - start) * 1000, 3)

        fragments = 0
        first_fragment_ms: float | None = None
        async for fragment in stream_answer(self._stream_generator, question, passages):
            if first_fragment_ms is None:
                first_fragment_ms = round((time.perf_counter() - start) * 1000, 3)
            fragments += 1
            yield fragment

        logger.info(
            "Answer streamed",
            extra={
                "n_sources": len(passages),
                "fragments": fragments,
                "retrieve_ms": retrieve_ms,
                # Time to first fragment is the number streaming exists to
                # improve; total latency alone cannot show it moved.
                "first_fragment_ms": first_fragment_ms,
                "total_ms": round((time.perf_counter() - start) * 1000, 3),
                "release_channel": self.settings.release_channel,
            },
        )

    # ----------------------------------------------------------- reindexing

    async def reindex(
        self,
        documents: list[dict[str, Any]] | None = None,
        source: Path | None = None,
        persist: bool = False,
        batch_size: int = REINDEX_BATCH_SIZE,
    ) -> ReindexResult:
        """Add documents to the live index without a restart.

        ``atomic=True``, so retrieval needs no lock: the engine swaps in an
        extended copy instead of mutating the index requests are searching.

        ``persist`` writes the grown index back to disk. Off by default --
        appending to the served index is a reversible in-memory change, while
        overwriting the artifact is not, and the artifact is DVC-tracked.
        """
        engine = self._require_engine()
        chunks = list(documents or [])
        if source is not None:
            chunks.extend(_read_documents(source))
        if not chunks:
            raise ValueError("no documents supplied: pass documents or a source path")

        result = await run_in_threadpool(
            lambda: engine.add_documents(chunks, batch_size=batch_size, atomic=True)
        )
        if persist and result.added:
            await run_in_threadpool(engine.save_index, self.settings.index_dir)
            logger.info(
                "Persisted the re-indexed artifact",
                extra={"index_dir": str(self.settings.index_dir)},
            )
        return result

    async def aclose(self) -> None:
        """Release the streaming generator's connection pool, if it has one."""
        closer = getattr(self._stream_generator, "aclose", None)
        if closer is not None:
            await closer()


def _read_documents(source: Path) -> list[dict[str, Any]]:
    """Load chunk dicts from a JSON file holding a list of them."""
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{source} must hold a JSON list of chunk objects")
    return payload
