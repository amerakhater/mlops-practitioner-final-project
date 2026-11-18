"""Runtime settings, read from the environment with a ``PRODLLM_`` prefix.

Two deliberate choices here:

``artifacts_dir`` defaults to a **relative** path. A hardcoded absolute path is
the handbook's "works on your laptop, fails in the container" failure mode; the
container overrides it with ``PRODLLM_ARTIFACTS_DIR=/app/artifacts``.

No field is named ``model_*``. Pydantic v2 reserves that namespace, so a field
called ``model_file`` or ``model_id`` warns or errors -- hence
``onnx_model_file`` here and ``source_model_id`` in the schemas.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the request-validation limits, so the Pydantic
# schemas and the Settings defaults cannot drift apart.
MAX_TEXT_CHARS = 8000
MAX_BATCH_SIZE = 64
MAX_QUESTION_CHARS = 2000


class Settings(BaseSettings):
    """Service configuration. Every field is overridable via ``PRODLLM_<NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="PRODLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifacts_dir: Path = Path("artifacts")
    # Named `weights_dirname`, not `model_dirname`: see the note above about
    # pydantic's reserved `model_` namespace.
    weights_dirname: str = "model"
    index_dirname: str = "index"
    metadata_file: str = "metadata.json"

    # The model is fetched from Hugging Face on first start when the weights
    # are not already present, so the image stays thin. Set
    # PRODLLM_ALLOW_DOWNLOAD=false for an air-gapped deployment with baked-in
    # weights, and startup will fail loudly instead of reaching out.
    source_model_id: str = "intfloat/multilingual-e5-small"
    allow_download: bool = True

    max_seq_length: int = 512
    max_batch_size: int = MAX_BATCH_SIZE
    max_text_chars: int = MAX_TEXT_CHARS

    # --- retrieval ---
    corpus_chunks_path: Path = Path("data/processed/chunks.json")
    top_k: int = 5
    # Cosine floor, disabled by default. On this corpus the on-topic and
    # off-topic score ranges overlap, so no absolute threshold separates them
    # -- see the measurements in prodllm.rag.DEFAULT_MIN_SCORE. Left
    # configurable so it can be raised once a reranker makes scores
    # discriminative.
    min_score: float = 0.0
    max_question_chars: int = MAX_QUESTION_CHARS

    # --- generation ---
    # Empty base URL means no LLM is configured and the extractive generator
    # is used, which quotes articles verbatim and cannot hallucinate.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0

    torch_num_threads: int = 1
    log_level: str = "INFO"
    eager_load: bool = True

    @property
    def weights_dir(self) -> Path:
        """Directory holding config.json, the tokenizer files and the weights."""
        return self.artifacts_dir / self.weights_dirname

    @property
    def index_dir(self) -> Path:
        """Directory holding the FAISS index and its metadata sidecar."""
        return self.artifacts_dir / self.index_dirname

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_file

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor, so the environment is parsed once per process."""
    return Settings()
