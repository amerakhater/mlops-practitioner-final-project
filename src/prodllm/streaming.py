"""Token-level streaming for the generation step.

``prodllm.rag`` generates an answer in one blocking call, which is the right
shape for the evaluation harness and for ``POST /ask``: a score needs the whole
answer, and so does a JSON response body. Streaming is a different contract --
the caller wants the first words as soon as they exist, not the last one sooner
-- so it lives here rather than being bolted onto ``Generator``.

Three things are deliberate:

* **The protocol is an async generator, not a callback.** ``astream`` yields
  text fragments, so backpressure is the language's problem rather than ours:
  if a client reads slowly, the ``async for`` in the service simply suspends.

* **The wire format is OpenAI-compatible SSE, which is what makes the backend a
  configuration choice.** OpenRouter and a self-hosted vLLM server speak the
  same ``/chat/completions`` stream, so switching between them is a change to
  ``PRODLLM_LLM_BASE_URL`` and nothing else. No vendor SDK is involved.

* **There is a streaming path that needs no credentials.** With no LLM
  configured the service streams the retrieved article text itself. That is not
  a mock: the bytes are real, they arrive incrementally, and -- as everywhere
  else in this project -- quoting the code verbatim cannot hallucinate. It is
  what lets the streaming endpoint, its tests and a load test all run offline.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

import httpx

from prodllm.rag import (
    NO_ARTICLE_FOUND,
    SYSTEM_PROMPT,
    ExtractiveGenerator,
    RetrievedArticle,
    build_prompt,
)

logger = logging.getLogger("prodllm.streaming")

# SSE framing. A line is either a comment (OpenRouter sends ": OPENROUTER
# PROCESSING" keep-alives, which are not JSON and must not be parsed) or a
# "data:" payload, terminated by the sentinel below.
_SSE_DATA_PREFIX = "data:"
_SSE_COMMENT_PREFIX = ":"
_SSE_DONE = "[DONE]"

# How the extractive stream is cut into fragments: runs of non-whitespace with
# their trailing whitespace kept, so reassembling the fragments reproduces the
# text exactly. Chunking by whitespace rather than by character keeps Arabic
# intact -- slicing mid-grapheme would split combining marks off their letter.
_WORD_RE = re.compile(r"\S+\s*")
_WORDS_PER_FRAGMENT = 4


class StreamGenerator(Protocol):
    """Yields an answer in fragments, in order, as it becomes available."""

    def astream(
        self, question: str, passages: Sequence[RetrievedArticle]
    ) -> AsyncIterator[str]: ...


def iter_fragments(text: str, words: int = _WORDS_PER_FRAGMENT) -> list[str]:
    """Cut ``text`` into fragments that concatenate back to ``text`` exactly.

    Exposed and tested separately because "the stream reassembles to the same
    answer the blocking path would have returned" is the property that matters,
    and it is far easier to assert on a pure function than on a live socket.
    """
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return []
    return ["".join(tokens[i : i + words]) for i in range(0, len(tokens), words)]


class ExtractiveStreamGenerator:
    """Streams the retrieved article text, with no LLM and no network.

    The default when no LLM is configured, mirroring ``ExtractiveGenerator`` in
    ``prodllm.rag``: the same text, delivered incrementally.

    ``delay_seconds`` exists for one narrow reason -- making the increments
    visible in ``curl``. Locally this text is assembled in well under a
    millisecond, so without a pause the fragments are genuinely streamed but
    arrive as a single visual blob, and a reviewer cannot tell streaming from
    buffering. It defaults to 0.0 and the measured numbers in reports/ are
    taken with it at 0.0; it is a demonstration aid, never a production value.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._delay_seconds = delay_seconds
        self._blocking = ExtractiveGenerator()

    async def astream(
        self, question: str, passages: Sequence[RetrievedArticle]
    ) -> AsyncIterator[str]:
        text = self._blocking.generate(question, passages)
        for fragment in iter_fragments(text):
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            yield fragment


class HttpStreamGenerator:
    """Streams from an OpenAI-compatible ``/chat/completions`` endpoint.

    One ``AsyncClient`` is shared across requests on purpose. A client per
    request would open a fresh TCP and TLS connection every time, which under
    the 50-user load test is both the dominant cost and a good way to exhaust
    ephemeral ports; the pool keeps connections warm instead.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_output_tokens: int = 700,
        max_connections: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``transport`` is the test seam.

        Passing an ``httpx.MockTransport`` lets the SSE loop be driven against
        a scripted stream -- including the keep-alive comments and the
        malformed chunk that only a live provider would otherwise produce -- so
        the streaming path is covered with no network and no API key.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        # connect/read/write/pool, generous on read: a slow first token is
        # normal for a loaded generator and must not look like a failure.
        self._timeout = httpx.Timeout(timeout, connect=10.0)
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=self._limits,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        """Release the connection pool. Called from the service's shutdown hook."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _payload(
        self, question: str, passages: Sequence[RetrievedArticle]
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, passages)},
            ],
        }

    @staticmethod
    def parse_sse_line(line: str) -> str | None:
        """Pull the text delta out of one SSE line, or ``None`` if it has none.

        Returns ``None`` for blank lines, keep-alive comments, the ``[DONE]``
        sentinel, unparseable JSON and the usage-only final chunk. Separated
        from the socket so every one of those cases is unit-testable; a stray
        keep-alive line reaching ``json.loads`` is the classic way a streaming
        client dies three minutes into a load test.
        """
        line = line.strip()
        if not line or line.startswith(_SSE_COMMENT_PREFIX):
            return None
        if not line.startswith(_SSE_DATA_PREFIX):
            return None
        payload = line[len(_SSE_DATA_PREFIX) :].strip()
        if not payload or payload == _SSE_DONE:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Discarding unparseable SSE payload")
            return None
        choices = parsed.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        # Only `content`. Reasoning models also emit `delta.reasoning`, which is
        # the model's scratchpad -- streaming it to a legal caller would present
        # unverified deliberation as part of the cited answer.
        return content if isinstance(content, str) and content else None

    async def astream(
        self, question: str, passages: Sequence[RetrievedArticle]
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=self._payload(question, passages),
        ) as response:
            if response.status_code >= 400:
                # The body carries the provider's reason, but a streaming
                # response has not read it yet, so raise_for_status() alone
                # would report the status and discard the explanation.
                body = (await response.aread()).decode("utf-8", "replace")
                logger.error(
                    "Generator rejected the request",
                    extra={"status_code": response.status_code, "body": body[:500]},
                )
                response.raise_for_status()
            async for line in response.aiter_lines():
                fragment = self.parse_sse_line(line)
                if fragment is not None:
                    yield fragment


def build_stream_generator(settings: Any) -> StreamGenerator:
    """Pick a streaming backend from configuration.

    Mirrors ``prodllm.rag.build_generator``, including its default: with no LLM
    configured the service streams verbatim article text rather than failing.
    ``llm_base_url`` is the only thing that distinguishes OpenRouter from a
    self-hosted vLLM server, because both speak the same stream.
    """
    if settings.llm_configured:
        logger.info(
            "Streaming from the HTTP generator",
            extra={"llm_model": settings.llm_model, "base_url": settings.llm_base_url},
        )
        return HttpStreamGenerator(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    logger.warning(
        "No LLM configured; streaming verbatim article text instead",
        extra={"demo_delay_ms": settings.stream_demo_delay_ms},
    )
    return ExtractiveStreamGenerator(
        delay_seconds=settings.stream_demo_delay_ms / 1000.0
    )


async def stream_answer(
    generator: StreamGenerator,
    question: str,
    passages: Sequence[RetrievedArticle],
) -> AsyncIterator[str]:
    """Stream an answer, short-circuiting when retrieval found nothing.

    The refusal is emitted here rather than sent to the generator: with no
    passages there is nothing to ground an answer in, so calling a paid
    endpoint to have it say "I don't know" costs money and invites it to
    improvise. The wording matches the blocking path exactly.
    """
    if not passages:
        yield NO_ARTICLE_FOUND
        return
    async for fragment in generator.astream(question, passages):
        yield fragment
