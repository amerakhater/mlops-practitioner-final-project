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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import faiss
import httpx
import numpy as np

from prodllm.corpus.normalize import normalize_arabic
from prodllm.embedding.base import Embedder
from prodllm.instrumentation import log_latency
from prodllm.rerank.base import Reranker

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

# How many deduplicated articles the first stage hands the re-ranker. 20 for
# 5 returned: deep enough that the cross-encoder has something to find below
# the vector cut, shallow enough that the per-request cost stays bounded --
# this is the number that multiplies into the re-ranking latency, because a
# cross-encoder is one forward pass per pair.
DEFAULT_RERANK_CANDIDATES = 20


def rerank_payload(article: "RetrievedArticle") -> str:
    """The passage text a cross-encoder scores the question against.

    Both languages, joined. The corpus is bilingual and the question may be in
    either language, so handing the re-ranker only one side would make it
    score an Arabic question against English text it was never going to match
    well -- the exact cross-lingual failure the two-vectors-per-chunk design in
    ``chunk_payloads`` exists to avoid at the first stage.

    The citation leads, so the model sees which article it is scoring. It is a
    short prefix and it is what the generator sees too, which keeps the two
    stages looking at the same thing.
    """
    parts = [f"[{article.citation}]"]
    if article.text_ar:
        parts.append(article.text_ar)
    if article.text_en:
        parts.append(article.text_en)
    return "\n".join(parts)


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
    # The cross-encoder's logit, or None when no re-ranker ran. Kept *beside*
    # ``score`` rather than replacing it, because the two are not the same
    # measurement and a client that has been reading ``score`` as a cosine
    # should not silently start receiving an unbounded logit instead. Having
    # both is also what makes the re-ranker auditable from a single response:
    # a large disagreement between the two orderings is the second stage
    # earning its latency, and a small one is it not.
    rerank_score: float | None = None

    def as_source(self) -> dict[str, Any]:
        """The shape returned in an API response's ``sources`` list."""
        return {
            "article_number": self.article_number,
            "citation": self.citation,
            "source_page": self.source_page,
            "score": round(self.score, 4),
            "rerank_score": (
                None if self.rerank_score is None else round(self.rerank_score, 4)
            ),
            "text_ar": self.text_ar,
            "text_en": self.text_en,
        }


@dataclass
class Answer:
    """A grounded answer plus the articles it was drawn from."""

    answer: str
    sources: list[RetrievedArticle] = field(default_factory=list)
    grounded: bool = True


# The generation contract. At module scope rather than inside HttpGenerator
# because two adapters now send it: the blocking HttpGenerator below and the
# streaming one in prodllm.streaming. A second copy of a prompt whose entire
# job is to forbid hallucination is exactly the kind of drift that stays
# invisible until an answer cites an article that does not exist.
SYSTEM_PROMPT = (
    "You answer questions about the Egyptian Civil Code using ONLY the "
    "articles provided. Cite the article number for every statement, using "
    "the citation strings given. If the provided articles do not answer the "
    "question, say so plainly and do not speculate. Never invent an article "
    "number or a provision. Answer in the language of the question."
)

# The refusal used when retrieval returns nothing, shared for the same reason:
# the extractive generator, /ask and the streaming path must decline in
# identical words, or a client cannot recognise a refusal without parsing prose.
NO_ARTICLE_FOUND = (
    "No article in the Egyptian Civil Code was found that addresses this question."
)


def build_prompt(question: str, passages: Sequence[RetrievedArticle]) -> str:
    """Render the retrieved articles and the question into the user turn."""
    context = "\n\n".join(
        f"[{p.citation}]\nArabic: {p.text_ar}\nEnglish: {p.text_en}" for p in passages
    )
    return f"Articles:\n{context}\n\nQuestion: {question}"


@dataclass(frozen=True)
class ReindexResult:
    """What a batch re-index did, as a value the caller can assert on.

    ``added`` counts documents, ``vectors_added`` counts rows in the index --
    they differ because a bilingual chunk contributes one vector per language,
    and conflating them is how "documents indexed" ends up reported at twice
    its real value.
    """

    added: int
    skipped: int
    vectors_added: int
    documents_indexed: int

    def as_dict(self) -> dict[str, int]:
        """The shape returned by the re-index endpoint and the CLI's --json."""
        return {
            "added": self.added,
            "skipped": self.skipped,
            "vectors_added": self.vectors_added,
            "documents_indexed": self.documents_indexed,
        }


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
            return NO_ARTICLE_FOUND
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

    # Both kept on the class because callers and tests reach for
    # HttpGenerator.SYSTEM_PROMPT and .build_prompt(). The definitions
    # themselves live at module scope so the streaming adapter in
    # prodllm.streaming shares them rather than keeping a second copy.
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_prompt(self, question: str, passages: Sequence[RetrievedArticle]) -> str:
        return build_prompt(question, passages)

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


def build_reranker(settings: Any) -> Reranker | None:
    """Pick a second retrieval stage from configuration, or None.

    Returns None in three cases, and all three are a working service rather
    than an error: re-ranking is switched off, the artifact is not on disk, or
    loading it failed. Vector-only retrieval is what this project served for
    three modules; downgrading to it is a measurable quality loss and is logged
    as a warning, but refusing to start would be worse.

    ``local_files_only`` is set when the source is a directory and cleared when
    it is a Hugging Face id. A directory is the baked-in artifact, and reaching
    out to the network for it would mean the image's provenance could not be
    verified -- the same argument as ``TorchEmbedder.from_settings``. An
    explicit model id, by contrast, is a deliberate request to fetch.
    """
    if not settings.reranker_enabled:
        logger.info("Re-ranking disabled; retrieval is vector-only")
        return None

    source = settings.reranker_source
    from_directory = not settings.reranker_model_id
    if from_directory and not Path(source).is_dir():
        logger.warning(
            "Re-ranking enabled but no re-ranker artifact found; "
            "falling back to vector-only retrieval",
            extra={"reranker_source": source},
        )
        return None

    try:
        from prodllm.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker.load(
            source,
            max_seq_length=settings.reranker_max_seq_length,
            batch_size=settings.reranker_batch_size,
            local_files_only=from_directory,
        )
    except Exception as exc:
        logger.warning(
            "Re-ranker failed to load; falling back to vector-only retrieval",
            extra={"reranker_source": source, "error": str(exc)},
        )
        return None


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
        return len(self.chunk_ids)

    @property
    def chunk_ids(self) -> set[Any]:
        """The identifier of every chunk in the index.

        Falls back to ``article_number`` for metadata written before chunks
        carried ids. This is what makes re-indexing idempotent: the batch job
        diffs against this set instead of trusting the caller to have filtered.
        """
        return {
            entry.get("chunk_id") or entry.get("article_number")
            for entry in self._metadata
        }

    @property
    def dim(self) -> int:
        """The embedding width this index was built at."""
        return int(self._index.d)

    def _prepare(
        self, vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> np.ndarray:
        """Validate an append and return the contiguous float32 matrix.

        The dimension is checked here rather than left to FAISS: a mismatch
        means the new documents were embedded with a different model, and
        FAISS' own failure for that is a bare assertion with no explanation.
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata ({len(metadata)}) "
                "must be the same length"
            )
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim mismatch: index is {self.dim}, new vectors are "
                f"{matrix.shape[1] if matrix.ndim == 2 else matrix.shape} -- the "
                "new documents were embedded with a different model, so the "
                "whole index needs rebuilding"
            )
        return matrix

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Append vectors to this index, in place.

        A flat index supports this directly: there is nothing trained to
        invalidate, so appending is exact and costs no recall -- which is part
        of why a flat index is still the right choice at this corpus size.

        In place, so this is for the offline job, where nothing is reading the
        index concurrently. A live service uses ``extended`` instead.
        """
        if not metadata:
            return
        self._index.add(self._prepare(vectors, metadata))
        self._metadata.extend(metadata)

    def extended(
        self, vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> "VectorIndex":
        """A **new** index holding this one's vectors plus ``vectors``.

        The copy-on-write half of re-indexing, and the reason the live service
        needs no lock on its read path. ``faiss.IndexFlatIP.add`` is not safe
        against a concurrent ``search``, so mutating the served index in place
        would mean either a race or a mutex that serialises every retrieval --
        and retrieval is the hot path under the 50-user load test.

        Building a copy and rebinding the attribute avoids both. Readers that
        already hold the old object finish against a consistent snapshot, and
        the swap is a single attribute assignment, so no request ever observes
        a half-populated index. The copy is cheap at this scale: the whole
        index is a few megabytes.
        """
        if not metadata:
            return VectorIndex(faiss.clone_index(self._index), list(self._metadata))
        matrix = self._prepare(vectors, metadata)
        clone = faiss.clone_index(self._index)
        clone.add(matrix)
        return VectorIndex(clone, [*self._metadata, *metadata])

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
        reranker: "Reranker | None" = None,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._generator: Generator = generator or ExtractiveGenerator()
        self._top_k = top_k
        self._min_score = min_score
        # Optional, and None is a first-class configuration rather than a
        # degraded one: the single-stage path is what every measurement before
        # Module 4 was taken against, and it has to stay runnable for the
        # before/after comparison to mean anything.
        self._reranker = reranker
        self._rerank_candidates = max(rerank_candidates, top_k)

    @property
    def reranker(self) -> "Reranker | None":
        """The second stage, or None when retrieval is vector-only."""
        return self._reranker

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

    # ------------------------------------------------------- batch reindexing

    @log_latency("rag.reindex")
    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
        atomic: bool = False,
    ) -> "ReindexResult":
        """Embed and append documents the index does not already hold.

        Incremental by design. Re-embedding ~1100 chunks to publish one new
        article takes minutes and, more importantly, rewrites vectors that
        nothing asked to change; appending touches only the new rows.

        **Idempotent.** Chunks whose id is already indexed are skipped, so the
        job can be re-run after a partial failure, or pointed at the whole
        corpus file, without duplicating anything. Duplicates would not merely
        waste space: ``retrieve`` dedupes by article number and keeps the
        best-scoring vector, so a doubled chunk silently consumes a slot in the
        candidate pool and pushes a genuine article out of the top-k.

        ``atomic`` selects how the new rows become visible. ``False`` appends
        in place, which is right for the offline job. ``True`` swaps in an
        extended copy, which is what a live service needs -- see
        ``VectorIndex.extended``.

        Known limitation: this adds, it does not revise. A flat index can
        delete by id, but doing so compacts the remaining rows and renumbers
        them, which would desynchronise the metadata list that positions index
        into. Amending an existing article therefore still needs a full
        rebuild, and that is what scripts/build_index.py is for.
        """
        known = self._index.chunk_ids
        fresh = [
            chunk
            for chunk in chunks
            if (chunk.get("chunk_id") or chunk.get("article_number")) not in known
        ]
        skipped = len(chunks) - len(fresh)
        if not fresh:
            logger.info(
                "Re-index found nothing new",
                extra={"skipped": skipped, "documents_indexed": self.documents_indexed},
            )
            return ReindexResult(
                added=0,
                skipped=skipped,
                vectors_added=0,
                documents_indexed=self.documents_indexed,
            )

        texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        for chunk in fresh:
            for language, text in self.chunk_payloads(chunk):
                texts.append(text)
                metadata.append({**chunk, "lang": language})

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(
                self._embedder.embed(
                    texts[start : start + batch_size], prefix="passage: "
                )
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        if atomic:
            # Rebind, do not mutate: a request already inside retrieve() keeps
            # the snapshot it started with, and the next one sees the new rows.
            self._index = self._index.extended(matrix, metadata)
        else:
            self._index.add(matrix, metadata)

        result = ReindexResult(
            added=len(fresh),
            skipped=skipped,
            vectors_added=len(metadata),
            documents_indexed=self.documents_indexed,
        )
        logger.info(
            "Re-indexed new documents",
            extra={
                "added": result.added,
                "skipped": result.skipped,
                "vectors_added": result.vectors_added,
                "documents_indexed": result.documents_indexed,
            },
        )
        return result

    def save_index(self, directory: Path) -> None:
        """Persist the current index, so an append survives a restart."""
        self._index.save(directory)

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
        #
        # With a re-ranker attached the pool is deeper again. The entire value
        # of a second stage is that it can promote an article the cosine
        # ranked below the cut, and it can only promote what it is shown --
        # re-ranking the same five articles the vector search already chose
        # would cost a forward pass per pair and change nothing.
        pool = self._top_k if self._reranker is None else self._rerank_candidates
        candidates = self._index.search(vector, pool * CANDIDATE_MULTIPLIER)

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
        if self._reranker is None:
            return ranked[: self._top_k]
        return self._rerank(question, ranked[: self._rerank_candidates])

    @log_latency("rag.rerank")
    def _rerank(
        self, question: str, candidates: list[RetrievedArticle]
    ) -> list[RetrievedArticle]:
        """Reorder ``candidates`` with the cross-encoder and keep the best.

        The raw question is scored, not the normalized form. Normalization
        strips the diacritics and unifies the hamza and alef variants that the
        embedding model is sensitive to; a cross-encoder trained on natural
        multilingual text is not, and feeding it stripped Arabic throws away
        signal for no benefit.

        A failing re-ranker degrades to the vector ordering rather than
        failing the request. The first-stage result is a genuine answer -- it
        is what this service returned for every request before Module 4 -- so
        turning a recoverable second-stage fault into a 500 would trade a
        slightly worse answer for no answer at all.
        """
        if not candidates:
            return []
        payloads = [rerank_payload(article) for article in candidates]
        try:
            order = self._reranker.rerank(question, payloads, top_k=self._top_k)
        except Exception:
            logger.exception(
                "Re-ranking failed; falling back to the vector ordering",
                extra={"n_candidates": len(candidates)},
            )
            return candidates[: self._top_k]

        reranked = [
            replace(candidates[position], rerank_score=float(score))
            for position, score in order
        ]
        logger.info(
            "Re-ranked candidates",
            extra={
                "n_candidates": len(candidates),
                "n_returned": len(reranked),
                "promoted": candidates[0].article_number != reranked[0].article_number,
            },
        )
        return reranked

    @log_latency("rag.answer")
    def answer(self, question: str) -> Answer:
        """Answer ``question``, grounded in retrieved articles."""
        passages = self.retrieve(question)
        if not passages:
            logger.info(
                "No relevant article found", extra={"question_chars": len(question)}
            )
            return Answer(answer=NO_ARTICLE_FOUND, sources=[], grounded=False)
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
