"""The streaming generation path.

No test here opens a socket or needs an API key. The HTTP adapter is driven
through an ``httpx.MockTransport`` carrying a scripted SSE stream, which is how
the cases a live provider produces only occasionally -- keep-alive comments, a
malformed chunk, a usage-only final frame -- become deterministic tests rather
than things discovered during a load run.

Async code is driven with ``asyncio.run`` inside ordinary sync tests rather
than with a pytest async plugin. That keeps the suite's dependencies as they
were and keeps ``--strict-markers`` meaningful: there is no new marker to
register, and nothing silently skips if a plugin is missing.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from prodllm.config import Settings
from prodllm.rag import NO_ARTICLE_FOUND, ExtractiveGenerator, RetrievedArticle
from prodllm.streaming import (
    ExtractiveStreamGenerator,
    HttpStreamGenerator,
    build_stream_generator,
    iter_fragments,
    stream_answer,
)


def collect(agen: AsyncIterator[str]) -> list[str]:
    """Drain an async generator into a list, from a synchronous test."""

    async def _drain() -> list[str]:
        return [fragment async for fragment in agen]

    return asyncio.run(_drain())


def article(
    number: int, text_en: str = "", text_ar: str = "نص عربي"
) -> RetrievedArticle:
    return RetrievedArticle(
        article_number=number,
        citation=f"Egyptian Civil Code, Article {number}",
        text_ar=text_ar,
        text_en=text_en or f"The text of article {number} in English.",
        score=0.9,
        source_page=1,
    )


# ------------------------------------------------------------- iter_fragments


@pytest.mark.parametrize(
    "text",
    [
        "one two three four five six seven eight nine ten",
        "short",
        "العقد شريعة المتعاقدين وفقا للمادة 147 من القانون المدني",
        "trailing whitespace   ",
        "multiple\n\nnewlines\nhere",
    ],
)
def test_fragments_reassemble_into_the_original_text(text):
    """The property that matters: streaming must not alter the answer.

    A client concatenating the stream has to end up with exactly what /ask
    would have returned, or the two endpoints disagree about what the law says.
    """
    assert "".join(iter_fragments(text)) == text


def test_fragments_of_empty_text_is_empty():
    assert iter_fragments("") == []
    assert iter_fragments("   ") == []


def test_fragments_are_actually_split():
    """Several fragments, not one -- otherwise 'streaming' is a single write."""
    fragments = iter_fragments("a b c d e f g h i j k l", words=4)

    assert len(fragments) == 3
    assert fragments[0] == "a b c d "


def test_arabic_fragments_keep_their_characters_intact():
    """Chunking is by whitespace, so no fragment boundary lands mid-grapheme."""
    text = "العقد شريعة المتعاقدين فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين"
    fragments = iter_fragments(text)

    assert len(fragments) > 1
    assert "".join(fragments) == text
    for fragment in fragments:
        assert fragment.strip()


# -------------------------------------------------- ExtractiveStreamGenerator


def test_extractive_stream_matches_the_blocking_generator():
    """The streamed answer and the buffered one must be the same text."""
    passages = [article(147), article(238)]
    expected = ExtractiveGenerator().generate("q", passages)

    streamed = "".join(collect(ExtractiveStreamGenerator().astream("q", passages)))

    assert streamed == expected


def test_extractive_stream_yields_more_than_one_fragment():
    fragments = collect(ExtractiveStreamGenerator().astream("q", [article(147)]))

    assert len(fragments) > 1


def test_extractive_stream_honours_the_demo_delay():
    """The delay is applied per fragment, so the stream is observably paced."""
    generator = ExtractiveStreamGenerator(delay_seconds=0.01)
    passages = [article(147)]

    async def _timed() -> float:
        loop = asyncio.get_running_loop()
        start = loop.time()
        async for _ in generator.astream("q", passages):
            pass
        return loop.time() - start

    fragments = len(iter_fragments(ExtractiveGenerator().generate("q", passages)))
    assert asyncio.run(_timed()) >= 0.01 * fragments * 0.5


# ----------------------------------------------------------- stream_answer


def test_stream_answer_refuses_in_the_same_words_as_the_blocking_path():
    """A refusal a client can recognise must be byte-identical across paths."""
    fragments = collect(stream_answer(ExtractiveStreamGenerator(), "q", []))

    assert "".join(fragments) == NO_ARTICLE_FOUND


def test_stream_answer_never_calls_the_generator_without_passages():
    """With nothing retrieved there is nothing to ground an answer in, so a
    paid endpoint must not be called to have it improvise a refusal."""

    class Exploding:
        def astream(self, question, passages):  # pragma: no cover - must not run
            raise AssertionError("the generator was called with no passages")

    assert "".join(collect(stream_answer(Exploding(), "q", []))) == NO_ARTICLE_FOUND


# ------------------------------------------------------------- SSE parsing


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('data: {"choices":[{"delta":{"content":"Hello"}}]}', "Hello"),
        ('data:{"choices":[{"delta":{"content":" world"}}]}', " world"),
        # OpenRouter sends SSE comments as keep-alives; they are not JSON.
        (": OPENROUTER PROCESSING", None),
        ("", None),
        ("   ", None),
        ("data: [DONE]", None),
        ("data:", None),
        ("event: message", None),
        ("data: not json at all", None),
        # The usage-only final frame has an empty choices list.
        ('data: {"choices":[],"usage":{"total_tokens":5}}', None),
        ('data: {"choices":[{"delta":{}}]}', None),
        ('data: {"choices":[{"delta":{"content":""}}]}', None),
        # Reasoning is the model's scratchpad, not part of the cited answer.
        ('data: {"choices":[{"delta":{"reasoning":"hmm"}}]}', None),
        ('data: {"choices":[{"delta":{"content":null}}]}', None),
    ],
)
def test_parse_sse_line(line, expected):
    assert HttpStreamGenerator.parse_sse_line(line) == expected


# --------------------------------------------------------- HttpStreamGenerator


def sse_stream(*contents: str, keepalives: bool = True) -> bytes:
    """Build an SSE body shaped like a real provider's, keep-alives included."""
    lines: list[str] = []
    for index, content in enumerate(contents):
        if keepalives and index == 1:
            lines.append(": OPENROUTER PROCESSING\n\n")
        frame = {"choices": [{"delta": {"content": content}, "index": 0}]}
        lines.append(f"data: {json.dumps(frame)}\n\n")
    lines.append('data: {"choices":[],"usage":{"total_tokens":9}}\n\n')
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def streaming_generator(body: bytes, status: int = 200) -> HttpStreamGenerator:
    def handler(request: httpx.Request) -> httpx.Response:
        handler.seen = request  # type: ignore[attr-defined]
        return httpx.Response(status, content=body)

    generator = HttpStreamGenerator(
        base_url="https://generator.invalid/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    generator.handler = handler  # type: ignore[attr-defined]
    return generator


def test_http_stream_yields_each_content_delta_in_order():
    generator = streaming_generator(
        sse_stream("The ", "term ", "is ", "fifteen years.")
    )

    fragments = collect(generator.astream("q", [article(374)]))

    assert fragments == ["The ", "term ", "is ", "fifteen years."]
    assert "".join(fragments) == "The term is fifteen years."


def test_http_stream_survives_keepalives_and_the_usage_frame():
    """A stray non-JSON keep-alive must not reach json.loads and kill the run."""
    generator = streaming_generator(sse_stream("a", "b", "c"))

    assert collect(generator.astream("q", [article(1)])) == ["a", "b", "c"]


def test_http_stream_skips_a_malformed_chunk_without_dropping_the_rest():
    body = (
        b'data: {"choices":[{"delta":{"content":"kept"}}]}\n\n'
        b"data: {truncated json\n\n"
        b'data: {"choices":[{"delta":{"content":" also kept"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    generator = streaming_generator(body)

    assert collect(generator.astream("q", [article(1)])) == ["kept", " also kept"]


def test_http_stream_sends_the_system_prompt_and_asks_for_a_stream():
    generator = streaming_generator(sse_stream("x"))

    collect(generator.astream("ما هو العقد؟", [article(147)]))

    sent = json.loads(generator.handler.seen.content)
    assert sent["stream"] is True
    assert sent["temperature"] == 0
    assert sent["messages"][0]["role"] == "system"
    assert "ONLY the" in sent["messages"][0]["content"]
    assert "Article 147" in sent["messages"][1]["content"]
    assert generator.handler.seen.headers["authorization"] == "Bearer test-key"


def test_http_stream_raises_on_an_error_status():
    generator = streaming_generator(b'{"error":"insufficient credit"}', status=402)

    with pytest.raises(httpx.HTTPStatusError):
        collect(generator.astream("q", [article(1)]))


def test_http_stream_closes_its_client():
    generator = streaming_generator(sse_stream("x"))
    collect(generator.astream("q", [article(1)]))

    asyncio.run(generator.aclose())

    assert generator._client is None


# ------------------------------------------------------ build_stream_generator


def test_build_stream_generator_defaults_to_extractive_without_an_llm():
    assert isinstance(build_stream_generator(Settings()), ExtractiveStreamGenerator)


def test_build_stream_generator_returns_the_http_adapter_when_configured():
    settings = Settings(
        llm_base_url="https://generator.invalid/v1",
        llm_model="test-model",
        llm_api_key="key",
    )

    assert isinstance(build_stream_generator(settings), HttpStreamGenerator)


def test_build_stream_generator_passes_the_demo_delay_through():
    settings = Settings(stream_demo_delay_ms=25)

    generator = build_stream_generator(settings)

    assert isinstance(generator, ExtractiveStreamGenerator)
    assert generator._delay_seconds == pytest.approx(0.025)
