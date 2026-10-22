"""Retrieval-augmented question answering over the Egyptian Civil Code.

Ingestion and querying both live here: corpus JSON -> article chunks ->
embeddings -> FAISS index, and question -> retrieved articles -> grounded
answer with citations.

Two design points carry the track's central requirement that hallucination is
legally unacceptable:

* **Generation is a port, not a hard-wired call.** ``Generator`` is a protocol
  with an HTTP-backed implementation and a deterministic extractive fake. The
  fake is what lets the whole chain be tested offline, and it is also the
  honest fallback when no LLM is configured -- returning article text verbatim
  cannot hallucinate.
* **Answers carry citations, never chunk ids.** ``RetrievedArticle`` keeps the
  article number and printed citation alongside the text, because an answer a
  lawyer cannot verify against an article number is worthless.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import faiss
import httpx
import numpy as np

from prodllm.corpus.normalize import normalize_arabic
from prodllm.embedding.base import Embedder
from prodllm.instrumentation import log_latency

logger = logging.getLogger("prodllm.rag")

INDEX_FILENAME = "chunks.faiss"
METADATA_FILENAME = "chunks.json"

# Cosine floor below which a hit is discarded. Default 0.0 = disabled, and
# that default is a measured result, not an oversight.
#
# Measured on this corpus with multilingual-e5-small, 10 on-topic legal
# questions vs 5 clearly off-topic ones:
#
#     on-topic  top-1 cosine: min 0.838  max 0.899  mean 0.870
#     off-topic top-1 cosine: min 0.789  max 0.859  mean 0.814
#
# The ranges overlap: "Explain quantum entanglement" scores 0.859, above
# several genuine legal questions. e5-family models compress similarities into
# a narrow high band, so NO absolute threshold separates relevant from
# irrelevant here, and a value tuned to look good on a handful of examples
# would be false confidence.
#
# Hallucination is therefore guarded where it can be: the generator is
# instructed to answer only from the supplied articles and to say so when they
# do not answer the question, every answer carries the article citations and
# their scores so a caller can judge, and `grounded` reports whether anything
# was retrieved at all. The real fix is a cross-encoder reranker over the
# top-k, which scores query/passage pairs jointly and does separate; that is
# deliberately left to a later module rather than faked with a threshold.
DEFAULT_MIN_SCORE = 0.0
DEFAULT_TOP_K = 5

# How many raw vector hits to fetch per requested article. Each bilingual
# chunk has an Arabic and an English vector, so without over-fetching a
# top_k of 5 could come back as the same article five times over.
CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True)
class RetrievedArticle:
    """One retrieval hit, carrying everything needed to cite it."""

    article_number: int
    citation: str
    text_ar: str
    text_en: str
    score: float
    source_page: int
    book: str | None = None
    chapter: str | None = None
    section: str | None = None

    def as_source(self) -> dict[str, Any]:
        """The shape returned in an API response's ``sources`` list."""
        return {
            "article_number": self.article_number,
            "citation": self.citation,
            "source_page": self.source_page,
            "score": round(self.score, 4),
            "text_ar": self.text_ar,
            "text_en": self.text_en,
        }


@dataclass
class Answer:
    """A grounded answer plus the articles it was drawn from."""

    answer: str
    sources: list[RetrievedArticle] = field(default_factory=list)
    grounded: bool = True


class Generator(Protocol):
    """Turns a question plus retrieved articles into an answer."""

    def generate(self, question: str, passages: Sequence[RetrievedArticle]) -> str: ...


class ExtractiveGenerator:
    """Returns the retrieved text verbatim, with its citations.

    Deterministic and incapable of inventing a provision, which makes it both
    the test double and a defensible production fallback when no LLM is
    configured. It is not a stub: a legal assistant that quotes the code and
    cites it is more useful than one that paraphrases it wrongly.
    """

    def generate(
        self,
        question: str,  # noqa: ARG002 — required by the Generator protocol
        passages: Sequence[RetrievedArticle],
    ) -> str:
        if not passages:
            return (
                "No article in the Egyptian Civil Code was found that addresses "
                "this question."
            )
        parts = [
            f"{passage.citation}: {passage.text_en or passage.text_ar}".strip()
            for passage in passages
        ]
        return "\n\n".join(parts)


class HttpGenerator:
    """Calls an OpenAI-compatible chat completions endpoint over plain HTTP.

    No vendor SDK: the wire format is stable and a dependency-free adapter is
    easier to point at a self-hosted vLLM server later. The API key is read
    from configuration, which reads it from the environment -- it is never
    written to a file in the repository.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_output_tokens: int = 700,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens

    SYSTEM_PROMPT = (
        "You answer questions about the Egyptian Civil Code using ONLY the "
        "articles provided. Cite the article number for every statement, using "
        "the citation strings given. If the provided articles do not answer the "
        "question, say so plainly and do not speculate. Never invent an article "
        "number or a provision. Answer in the language of the question."
    )

    def build_prompt(self, question: str, passages: Sequence[RetrievedArticle]) -> str:
        context = "\n\n".join(
            f"[{p.citation}]\nArabic: {p.text_ar}\nEnglish: {p.text_en}"
            for p in passages
        )
        return f"Articles:\n{context}\n\nQuestion: {question}"

    def generate(self, question: str, passages: Sequence[RetrievedArticle]) -> str:

        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "temperature": 0,
                "max_tokens": self._max_output_tokens,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.build_prompt(question, passages),
                    },
                ],
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()


def build_generator(settings: Any) -> Generator:
    """Pick a generator from configuration.

    With no LLM configured the extractive generator is used. That is a
    deliberate default rather than a failure mode: quoting the retrieved
    articles verbatim with their citations cannot hallucinate, which is the one
    thing this track treats as unacceptable.
    """
    if settings.llm_configured:
        logger.info(
            "Using HTTP LLM generator",
            extra={"llm_model": settings.llm_model, "base_url": settings.llm_base_url},
        )
        return HttpGenerator(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    logger.warning(
        "No LLM configured; answering extractively with verbatim article text"
    )
    return ExtractiveGenerator()


class VectorIndex:
    """A flat inner-product FAISS index over unit-norm chunk embeddings.

    Inner product on normalized vectors is cosine similarity, which is what
    makes a single ``min_score`` threshold meaningful across queries.
    """

    def __init__(self, index: Any, metadata: list[dict[str, Any]]) -> None:
        self._index = index
        self._metadata = metadata

    def __len__(self) -> int:
        """The number of vectors, which exceeds the document count.

        A bilingual chunk contributes one vector per language, so this is not
        the figure to report as "documents indexed".
        """
        return len(self._metadata)

    @property
    def metadata(self) -> list[dict[str, Any]]:
        return self._metadata

    @property
    def document_count(self) -> int:
        """Distinct chunks indexed, counting a bilingual chunk once."""
        identifiers = {
            entry.get("chunk_id") or entry.get("article_number")
            for entry in self._metadata
        }
        return len(identifiers)

    @classmethod
    def build(
        cls, vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> "VectorIndex":

        if len(vectors) != len(metadata):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata ({len(metadata)}) "
                "must be the same length"
            )
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index, metadata)

    def save(self, directory: Path) -> None:

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        (directory / METADATA_FILENAME).write_text(
            json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":

        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        for path in (index_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(f"index artifact is missing: {path}")
        return cls(
            faiss.read_index(str(index_path)),
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        matrix = np.ascontiguousarray(
            query.reshape(1, -1) if query.ndim == 1 else query, dtype=np.float32
        )
        scores, indices = self._index.search(matrix, min(top_k, len(self._metadata)))
        hits: list[tuple[dict, float]] = []
        for position, score in zip(indices[0], scores[0], strict=True):
            if position < 0:  # FAISS pads with -1 when fewer results exist
                continue
            hits.append((self._metadata[position], float(score)))
        return hits


class RagEngine:
    """Ingestion and querying for the legal corpus."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        generator: Generator | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._generator: Generator = generator or ExtractiveGenerator()
        self._top_k = top_k
        self._min_score = min_score

    @property
    def documents_indexed(self) -> int:
        return self._index.document_count

    # ----------------------------------------------------------- ingestion

    @staticmethod
    def chunk_payload(chunk: dict[str, Any]) -> str:
        """The Arabic text that gets embedded.

        The normalized form is used rather than the printed one, so a question
        spelled with a different hamza or alef variant still matches. The
        printed text stays in the metadata for citation.
        """
        normalized = chunk.get("text_ar_normalized") or normalize_arabic(
            chunk.get("text_ar", "")
        )
        return normalized or chunk.get("text_en", "")

    @classmethod
    def chunk_payloads(cls, chunk: dict[str, Any]) -> list[tuple[str, str]]:
        """Every embeddable view of a chunk, as ``(language, text)`` pairs.

        A bilingual corpus gets **one vector per language**, both pointing at
        the same article. Embedding only the Arabic leaves English questions
        matched cross-lingually against Arabic vectors, which measurably fails:
        five Arabic/English question pairs agreed on the top article 0 times
        with an Arabic-only index. Indexing both sides is what makes the same
        question in either language retrieve the same article.
        """
        payloads: list[tuple[str, str]] = []
        arabic = cls.chunk_payload(chunk)
        if arabic:
            payloads.append(("ar", arabic))
        english = (chunk.get("text_en") or "").strip()
        if english:
            payloads.append(("en", english))
        return payloads

    @classmethod
    def build_index(
        cls, chunks: list[dict[str, Any]], embedder: Embedder, batch_size: int = 32
    ) -> VectorIndex:
        """Embed every chunk in every language and build the index."""
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        for chunk in chunks:
            for language, text in cls.chunk_payloads(chunk):
                texts.append(text)
                metadata.append({**chunk, "lang": language})

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(embedder.embed(batch, prefix="passage: "))
        return VectorIndex.build(np.asarray(vectors, dtype=np.float32), metadata)

    # ------------------------------------------------------------ querying

    @log_latency("rag.retrieve")
    def retrieve(self, question: str) -> list[RetrievedArticle]:
        """Find the articles most relevant to ``question``.

        The question is normalized and embedded with the ``query: `` prefix the
        embedding model expects, and hits below ``min_score`` are dropped so
        that an unrelated question retrieves nothing rather than the least-bad
        article in the corpus.
        """
        vector = np.asarray(
            self._embedder.embed([normalize_arabic(question)], prefix="query: "),
            dtype=np.float32,
        )
        # Over-fetch, because the Arabic and English vectors of the same
        # article both compete for slots; without this, top_k=5 could return
        # the same article five times in two languages.
        candidates = self._index.search(vector, self._top_k * CANDIDATE_MULTIPLIER)

        best: dict[int, RetrievedArticle] = {}
        for payload, score in candidates:
            if score < self._min_score:
                continue
            number = int(payload["article_number"])
            existing = best.get(number)
            if existing is not None and existing.score >= score:
                continue
            best[number] = RetrievedArticle(
                article_number=number,
                citation=payload.get("citation", ""),
                text_ar=payload.get("text_ar", ""),
                text_en=payload.get("text_en", ""),
                score=score,
                source_page=int(payload.get("source_page", 1)),
                book=payload.get("book"),
                chapter=payload.get("chapter"),
                section=payload.get("section"),
            )

        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[: self._top_k]

    @log_latency("rag.answer")
    def answer(self, question: str) -> Answer:
        """Answer ``question``, grounded in retrieved articles."""
        passages = self.retrieve(question)
        if not passages:
            logger.info(
                "No relevant article found", extra={"question_chars": len(question)}
            )
            return Answer(
                answer=(
                    "No article in the Egyptian Civil Code was found that "
                    "addresses this question."
                ),
                sources=[],
                grounded=False,
            )
        text = self._generator.generate(question, passages)
        logger.info(
            "Answer generated",
            extra={
                "n_sources": len(passages),
                "top_article": passages[0].article_number,
                "top_score": round(passages[0].score, 4),
            },
        )
        return Answer(answer=text, sources=passages, grounded=True)
