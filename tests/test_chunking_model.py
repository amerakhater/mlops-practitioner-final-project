"""The chunking configuration as a registered MLflow model.

Skipped unless the `tracking` group is installed. The default sync does not
include mlflow -- the service must never need it -- so this suite is
conditional in exactly the way the dependency is.
"""

import pytest

pytest.importorskip("mlflow", reason="requires the tracking dependency group")
pytest.importorskip("pandas", reason="requires the tracking dependency group")

import pandas as pd

from prodllm.corpus.chunk import ChunkConfig, chunk_corpus_with
from prodllm.corpus.schema import Article
from prodllm.tracking.chunking_model import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    ChunkingModel,
    example_input,
)


def frame(*articles: dict) -> pd.DataFrame:
    return pd.DataFrame(list(articles))


def article_row(number: int, text_ar: str = "نص المادة.", **overrides) -> dict:
    return {
        "article_number": number,
        "text_ar": text_ar,
        "text_en": "English",
        "source_page": 1,
        "is_repealed": False,
        **overrides,
    }


def test_predict_turns_articles_into_the_configured_chunks():
    model = ChunkingModel(ChunkConfig(name="baseline"))

    output = model.predict(None, frame(article_row(374), article_row(375)))

    assert list(output["article_number"]) == [374, 375]
    assert list(output["citation"]) == [
        "Egyptian Civil Code, Article 374",
        "Egyptian Civil Code, Article 375",
    ]


def test_predict_matches_the_chunker_it_wraps():
    config = ChunkConfig(name="narrow", chunk_size=300, overlap=50)
    text = " ".join(f"جملة رقم {n}." for n in range(120))
    rows = [article_row(1, text), article_row(2, text)]

    output = ChunkingModel(config).predict(None, frame(*rows))
    direct = chunk_corpus_with([Article.model_validate(row) for row in rows], config)

    assert list(output["chunk_id"]) == [chunk.chunk_id for chunk in direct]
    assert list(output["text_ar"]) == [chunk.text_ar for chunk in direct]


def test_two_configurations_chunk_the_same_articles_differently():
    text = " ".join(f"جملة رقم {n}." for n in range(200))
    rows = frame(article_row(1, text))

    wide = ChunkingModel(ChunkConfig(name="wide", chunk_size=4000)).predict(None, rows)
    narrow = ChunkingModel(ChunkConfig(name="narrow", chunk_size=300)).predict(
        None, rows
    )

    assert len(wide) == 1
    assert len(narrow) > 1


def test_a_repealed_article_produces_no_rows():
    model = ChunkingModel(ChunkConfig(name="baseline"))

    output = model.predict(
        None, frame(article_row(55, text_ar="", is_repealed=True), article_row(56))
    )

    assert list(output["article_number"]) == [56]


def test_an_empty_frame_yields_an_empty_frame_with_the_declared_columns():
    model = ChunkingModel(ChunkConfig(name="baseline"))

    output = model.predict(None, frame())

    assert len(output) == 0
    assert list(output.columns) == [
        "chunk_id",
        "article_number",
        "part",
        "parts_total",
        "text_ar",
        "citation",
    ]


def test_the_logged_example_is_something_the_model_can_actually_predict():
    # An input_example that does not round-trip makes `mlflow models predict`
    # fail on the registered version, which is the one thing it exists to show.
    output = ChunkingModel(ChunkConfig(name="baseline")).predict(None, example_input())

    assert len(output) == 1
    assert output.iloc[0]["article_number"] == 374


def test_the_registry_names_are_stable():
    # register_best.py and the report both name these; a rename that misses one
    # leaves an alias pointing at a model nothing refers to.
    assert REGISTERED_MODEL_NAME == "rag-chunking-config"
    assert PRODUCTION_ALIAS == "Production"
