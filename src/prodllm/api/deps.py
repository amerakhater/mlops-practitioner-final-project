"""Dependency wiring.

``get_embedder`` is defined *in terms of* ``get_optional_embedder`` so that a
test overriding only the optional one controls both -- FastAPI applies
``dependency_overrides`` at every node of the dependency graph, sub-dependencies
included. That single seam is what lets the whole API be tested without the
448 MB artifact.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request, status

from prodllm.config import Settings
from prodllm.embedding.base import Embedder

if TYPE_CHECKING:
    from prodllm.rag import RagEngine

_UNAVAILABLE = "embedding model is not loaded; the service is running degraded"
_INDEX_UNAVAILABLE = (
    "the legal corpus index is not loaded; run scripts/build_index.py and "
    "restart, or check PRODLLM_ARTIFACTS_DIR"
)


def get_optional_embedder(request: Request) -> Embedder | None:
    """The loaded embedder, or ``None`` when startup could not load one."""
    return getattr(request.app.state, "embedder", None)


def get_embedder(
    embedder: Annotated[Embedder | None, Depends(get_optional_embedder)],
) -> Embedder:
    """The loaded embedder, or a 503 for the endpoints that cannot work without it."""
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE
        )
    return embedder


def get_optional_engine(request: Request) -> "RagEngine | None":
    """The RAG engine, or ``None`` when the index could not be loaded."""
    return getattr(request.app.state, "engine", None)


def get_engine(
    engine: Annotated["RagEngine | None", Depends(get_optional_engine)],
) -> "RagEngine":
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_INDEX_UNAVAILABLE
        )
    return engine


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


OptionalEmbedderDep = Annotated[Embedder | None, Depends(get_optional_embedder)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
OptionalEngineDep = Annotated["RagEngine | None", Depends(get_optional_engine)]
EngineDep = Annotated["RagEngine", Depends(get_engine)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
