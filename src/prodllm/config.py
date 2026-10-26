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

    # --- re-ranking (Module 4) ---
    # Off by default, and that is not timidity. Every retrieval and latency
    # number recorded before Module 4 was taken single-stage, so the
    # before/after comparison needs the old path to stay exactly reachable;
    # and a cross-encoder that is missing from the artifacts directory must
    # degrade to vector-only rather than fail startup, because vector-only is
    # a working service and not an error state.
    reranker_enabled: bool = False
    # Subdirectory of ARTIFACTS_DIR holding the distilled student, as written
    # by scripts/distill_reranker.py. A Hugging Face model id also works,
    # which is what makes an A/B against the undistilled teacher a config
    # change rather than a code change.
    reranker_dirname: str = "reranker"
    reranker_model_id: str = ""
    # Candidates handed to the second stage. 20 is a measured choice: on this
    # corpus the gold article is inside the top 20 vector hits for 90% of the
    # question set and inside the top 5 for only 75%, so this is the number
    # that sets the re-ranker's ceiling. See reports/module-4.md.
    rerank_candidates: int = 20
    reranker_max_seq_length: int = 512
    reranker_batch_size: int = 8

    # --- generation ---
    # Empty base URL means no LLM is configured and the extractive generator
    # is used, which quotes articles verbatim and cannot hallucinate.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0

    # --- streaming (Module 3) ---
    # Pause between fragments on the extractive stream, so the increments are
    # visible in curl rather than arriving as one visual blob. A demonstration
    # aid only: it defaults to 0 and every measured number in reports/ is taken
    # at 0. It has no effect once an LLM is configured, where the pacing is the
    # generator's.
    stream_demo_delay_ms: int = 0

    # --- release identity (Module 3) ---
    # Which rollout slot this replica belongs to. Returned on /ask and attached
    # to every request log line, so during a canary rollout a latency or error
    # spike can be attributed to a variant instead of guessed at. Free-form by
    # design: the orchestrator owns the naming, not the application.
    release_channel: str = "stable"

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
    def reranker_dir(self) -> Path:
        """Directory holding the distilled cross-encoder."""
        return self.artifacts_dir / self.reranker_dirname

    @property
    def reranker_source(self) -> str:
        """Where to load the re-ranker from: an explicit id wins, else the dir.

        An explicit ``RERANKER_MODEL_ID`` takes precedence so that pointing the
        service at the 568M teacher for a comparison needs no file moves.
        """
        return self.reranker_model_id or str(self.reranker_dir)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor, so the environment is parsed once per process."""
    return Settings()
