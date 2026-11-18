"""Request and response models.

The validation limits come from ``prodllm.config`` so the schemas and the
settings defaults cannot drift apart.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prodllm.config import MAX_BATCH_SIZE, MAX_QUESTION_CHARS, MAX_TEXT_CHARS

# e5 is trained with instruction prefixes; a Literal makes an unsupported one a
# 422 at the edge rather than a quietly worse embedding.
Prefix = Literal["query: ", "passage: "]

_ARABIC_QUERY_EXAMPLE = "ما هي مدة التقادم في القانون المدني؟"


class HealthResponse(BaseModel):
    """Liveness. Returns 200 even when degraded; readiness is ``documents_indexed``."""

    # `model_loaded` collides with pydantic's reserved `model_` namespace. The
    # name is the clearer API, so the namespace guard is disabled rather than
    # the field renamed to something vaguer.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["healthy", "degraded"]
    documents_indexed: int
    version: str
    model_loaded: bool
    uptime_seconds: float


class SourceArticle(BaseModel):
    """A cited article behind an answer.

    The article number and its printed citation are what make an answer
    verifiable; a chunk id or a bare page number would not be.
    """

    article_number: int
    citation: str
    source_page: int
    score: float
    text_ar: str
    text_en: str


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        examples=["ما هي مدة التقادم في القانون المدني؟"],
    )

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace only")
        return value


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceArticle]
    grounded: bool = Field(
        description="False when no article passed the relevance threshold."
    )
    latency_ms: float


class MetadataResponse(BaseModel):
    service_version: str
    source_model_id: str
    embedding_dim: int
    max_seq_length: int
    serialization_format: str = Field(
        description="How the loaded weights were stored on disk: safetensors or pickle."
    )
    artifact_sha256: str
    pooling: str
    normalized: bool
    supported_prefixes: list[str]
    # Part of the API contract a client needs in order to batch correctly,
    # rather than discovering the limits through 422s.
    max_batch_size: int
    max_text_chars: int


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1, max_length=MAX_TEXT_CHARS, examples=[_ARABIC_QUERY_EXAMPLE]
    )
    prefix: Prefix = "query: "


class PredictResponse(BaseModel):
    embedding: list[float]
    dim: int
    source_model_id: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    prefix: Prefix = "query: "

    @field_validator("texts")
    @classmethod
    def _reject_blank_and_oversized_items(cls, value: list[str]) -> list[str]:
        if any(not text.strip() for text in value):
            raise ValueError("texts must not contain empty or whitespace-only strings")
        if any(len(text) > MAX_TEXT_CHARS for text in value):
            raise ValueError(f"each text must be at most {MAX_TEXT_CHARS} characters")
        return value


class BatchPredictResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    source_model_id: str
    latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
