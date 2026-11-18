"""The winning chunking configuration, as a registered MLflow model.

A chunking configuration is three numbers, and it would be easy to "register"
it by writing those three numbers into a tag. That would be a record of a
decision, not an artifact of it: nothing would stop params.yaml drifting away
from the registered values, and nothing downstream could *run* the registered
thing to check.

So the registered model is the chunker itself. ``predict`` takes article
records and returns the chunks this configuration produces from them, which
means a reviewer can load version 3 from the registry and reproduce exactly
the chunk boundaries that scored whatever version 3 scored.

The embedding model id rides along in the model's parameters rather than in
its code: the chunker does not embed anything, but the three values are one
configuration -- a chunking that was tuned against e5-base is not the same
configuration when it is served with e5-small.
"""

from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from prodllm.corpus.chunk import ChunkConfig, chunk_corpus_with
from prodllm.corpus.schema import Article

REGISTERED_MODEL_NAME = "rag-chunking-config"
PRODUCTION_ALIAS = "Production"

# Only the fields the chunker actually reads. A wider signature would promise
# callers that the model consumes a full Article record, which it does not.
_INPUT_SCHEMA = Schema(
    [
        ColSpec("long", "article_number"),
        ColSpec("string", "text_ar"),
        ColSpec("string", "text_en"),
        ColSpec("long", "source_page"),
        ColSpec("boolean", "is_repealed"),
    ]
)
_OUTPUT_SCHEMA = Schema(
    [
        ColSpec("string", "chunk_id"),
        ColSpec("long", "article_number"),
        ColSpec("long", "part"),
        ColSpec("long", "parts_total"),
        ColSpec("string", "text_ar"),
        ColSpec("string", "citation"),
    ]
)

SIGNATURE = ModelSignature(inputs=_INPUT_SCHEMA, outputs=_OUTPUT_SCHEMA)


class ChunkingModel(mlflow.pyfunc.PythonModel):
    """Applies one :class:`ChunkConfig` to a frame of article records."""

    def __init__(self, config: ChunkConfig) -> None:
        self.config = config

    def predict(
        self,
        context: Any,  # noqa: ARG002 — required by the PythonModel interface
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        articles = [
            Article.model_validate(record)
            for record in model_input.to_dict(orient="records")
        ]
        chunks = chunk_corpus_with(articles, self.config)
        return pd.DataFrame(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "article_number": chunk.article_number,
                    "part": chunk.part,
                    "parts_total": chunk.parts_total,
                    "text_ar": chunk.text_ar,
                    "citation": chunk.citation,
                }
                for chunk in chunks
            ],
            columns=[spec.name for spec in _OUTPUT_SCHEMA.inputs],
        )


def example_input() -> pd.DataFrame:
    """A one-article frame.

    Logged with the model so the signature is demonstrable, not merely
    declared: ``mlflow models predict`` on the registered version works
    without the caller having to guess the column names.
    """
    return pd.DataFrame(
        [
            {
                "article_number": 374,
                "text_ar": "تتقادم الالتزامات بخمس عشرة سنة فيما عدا الحالات الخاصة.",
                "text_en": (
                    "The term of prescription for obligations is fifteen years."
                ),
                "source_page": 50,
                "is_repealed": False,
            }
        ]
    )


def log_chunking_model(
    config: ChunkConfig,
    artifact_path: str = "chunking_model",
    registered_model_name: str | None = None,
) -> Any:
    """Log the chunker as a pyfunc model inside the active run."""
    return mlflow.pyfunc.log_model(
        name=artifact_path,
        python_model=ChunkingModel(config),
        signature=SIGNATURE,
        input_example=example_input(),
        registered_model_name=registered_model_name,
        # prodllm must be importable wherever the model is loaded; it is a
        # real installed package, so the dependency is named rather than the
        # source being copied into the artifact.
        pip_requirements=["pydantic>=2", "pandas", f"mlflow=={mlflow.__version__}"],
    )
