"""The Bento: the Module 3 serving surface.

    bentoml serve prodllm.serving.service:LegalRagService

Endpoints, and why each exists:

``POST /ask``         asynchronous, buffered. One JSON object with the answer
                      and its citations. What a client that needs the whole
                      answer before acting should call.
``POST /ask_stream``  the same answer, streamed as plain text as it is
                      produced, so ``curl -N`` prints it incrementally.
``POST /reindex``     batch re-indexing against the live index, no restart.
``POST /status``      documents indexed and which release channel answered.
                      Liveness probes stay on BentoML's own ``/readyz``.

This module is intentionally thin. Every decorated method parses its arguments,
delegates to ``RagServiceCore`` and shapes the result; the behaviour under test
lives in ``engine.py``, which imports no bentoml and needs no running server.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import bentoml
from annotated_types import Ge, Le
from pydantic import Field

import prodllm
from prodllm.api.schemas import SourceArticle
from prodllm.config import MAX_QUESTION_CHARS, get_settings
from prodllm.logging_conf import configure_logging
from prodllm.serving.engine import (
    REINDEX_BATCH_SIZE,
    RagServiceCore,
    ServiceUnavailable,
)

logger = logging.getLogger("prodllm.serving")

_settings = get_settings()

# Concurrency the replica advertises, and the single most consequential number
# in this file. The work behind /ask is blocking CPU handed to a threadpool, so
# past the point where every core is busy, extra concurrency converts into
# queueing delay rather than throughput -- it inflates p95 without serving one
# more request per second. 64 is above the 50-user load test on purpose: the
# test is meant to measure this service's latency under its target load, not to
# be throttled by an admission limit and measure the limit instead.
_CONCURRENCY = 64

_QuestionField = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="A question about the Egyptian Civil Code, in Arabic or English.",
        examples=["ما هي مدة التقادم في القانون المدني؟"],
    ),
]


@bentoml.service(
    name="prodllm-rag",
    traffic={
        # Generous: a streamed answer from a loaded generator legitimately
        # takes tens of seconds, and a timeout that cuts it off would look
        # like a service fault rather than a slow model.
        "timeout": 180,
        "concurrency": _CONCURRENCY,
    },
    # One worker. The encoder is ~450 MB resident, so each additional worker
    # costs that again, and PRODLLM_TORCH_NUM_THREADS=1 keeps a single worker
    # from fighting itself for cores. Scale with replicas, which is also what
    # the canary configuration in the README assumes.
    workers=1,
    resources={"cpu": "4"},
    metrics={"enabled": True},
    labels={
        "owner": "prodllm",
        "module": "3-serving",
        "release_channel": _settings.release_channel,
    },
)
class LegalRagService:
    """Retrieval-augmented Q&A over the Egyptian Civil Code."""

    def __init__(self) -> None:
        configure_logging(
            level=logging.getLevelNamesMapping().get(
                _settings.log_level.upper(), logging.INFO
            )
        )
        self.core = RagServiceCore(settings=_settings)

    @bentoml.on_shutdown
    async def _shutdown(self) -> None:
        """Close the generator's connection pool on the way down."""
        await self.core.aclose()

    @bentoml.api(route="/status")
    def status(self) -> dict[str, Any]:
        """Readiness detail and release identity.

        Named ``/status``, not ``/healthz``: BentoML already serves ``/healthz``,
        ``/livez`` and ``/readyz`` itself. Taking that name would have left a
        route where GET reaches BentoML's probe and POST reaches this method --
        two different answers at one path, which is a trap for whoever wires
        the next probe.

        So the orchestrator keeps using ``/readyz``, and this endpoint carries
        what a probe has no room for: how many documents are actually indexed
        and which release channel answered. 200 even when degraded, matching
        the FastAPI app -- readiness is ``documents_indexed``, not the status
        code, because a container whose only fault is a missing artifact should
        report that rather than crash-loop before it can.
        """
        return self.core.health()

    @bentoml.api(route="/ask")
    async def ask(self, question: _QuestionField) -> dict[str, Any]:
        """Answer a question, with the articles the answer is drawn from.

        Asynchronous: the handler awaits while the blocking retrieval and
        generation run in a threadpool, so one slow answer does not stall the
        others in flight.
        """
        try:
            result, latency_ms = await self.core.answer(question)
        except ServiceUnavailable as exc:
            raise bentoml.exceptions.ServiceUnavailable(str(exc)) from exc

        return {
            "answer": result.answer,
            "sources": [
                SourceArticle(**source.as_source()).model_dump()
                for source in result.sources
            ],
            "grounded": result.grounded,
            "latency_ms": latency_ms,
            "release_channel": self.core.settings.release_channel,
            "service_version": prodllm.__version__,
        }

    @bentoml.api(route="/ask_stream")
    async def ask_stream(self, question: _QuestionField) -> AsyncGenerator[str, None]:
        """Stream the answer as it is produced.

        Yields ``str``, which BentoML turns into a ``StreamingResponse``, so
        ``curl -N`` prints fragments as they arrive rather than buffering to
        the end. The citations are not in the stream -- the body is text, not
        JSON -- so a client that needs them calls ``/ask``; this endpoint is
        for showing a reader the answer sooner.
        """
        try:
            async for fragment in self.core.astream(question):
                yield fragment
        except ServiceUnavailable as exc:
            raise bentoml.exceptions.ServiceUnavailable(str(exc)) from exc

    @bentoml.api(route="/reindex")
    async def reindex(
        self,
        documents: list[dict[str, Any]],
        persist: bool = False,
        batch_size: Annotated[int, Ge(1), Le(256)] = REINDEX_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Add new documents to the live index, without a restart.

        Idempotent: a document whose chunk id is already indexed is counted in
        ``skipped`` and not embedded again, so re-posting the same batch after
        a timeout cannot duplicate it.
        """
        try:
            result = await self.core.reindex(
                documents=documents, persist=persist, batch_size=batch_size
            )
        except ServiceUnavailable as exc:
            raise bentoml.exceptions.ServiceUnavailable(str(exc)) from exc
        except ValueError as exc:
            raise bentoml.exceptions.InvalidArgument(str(exc)) from exc
        return result.as_dict()
