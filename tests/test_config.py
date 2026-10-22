"""Settings, including the guard against the handbook's hardcoded-paths failure mode."""

from pathlib import Path

import pytest

from prodllm.config import MAX_BATCH_SIZE, MAX_TEXT_CHARS, Settings, get_settings


def test_defaults_are_sane():
    settings = Settings()

    assert settings.artifacts_dir == Path("artifacts")
    assert settings.weights_dirname == "model"
    assert settings.max_seq_length == 512
    assert settings.max_batch_size == MAX_BATCH_SIZE
    assert settings.max_text_chars == MAX_TEXT_CHARS
    assert settings.log_level == "INFO"


def test_artifacts_dir_default_is_relative():
    """Works on your laptop, fails in the container -- every time.

    The default must never be an absolute path; the container overrides it with
    PRODLLM_ARTIFACTS_DIR=/app/artifacts.
    """
    assert not Settings().artifacts_dir.is_absolute()


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("PRODLLM_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("PRODLLM_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PRODLLM_MAX_BATCH_SIZE", "8")

    settings = Settings()

    assert settings.artifacts_dir == tmp_path
    assert settings.log_level == "DEBUG"
    assert settings.max_batch_size == 8


def test_derived_paths_sit_under_the_artifacts_dir(tmp_path: Path):
    settings = Settings(artifacts_dir=tmp_path)

    assert settings.weights_dir == tmp_path / settings.weights_dirname
    assert settings.metadata_path == tmp_path / settings.metadata_file


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_no_field_uses_pydantics_reserved_model_namespace():
    """A field named model_* would warn or error under pydantic v2, which is why
    the settings use onnx_model_file and the schemas use source_model_id."""
    offenders = [n for n in Settings.model_fields if n.startswith("model_")]

    assert offenders == [], f"reserved namespace: {offenders}"
