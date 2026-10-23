"""The committed contract must describe the code that is actually shipped.

A stale openapi.json is worse than none: a client generated from it compiles
and then fails at runtime against a route that no longer looks like that. The
pre-commit hook rewrites the file, but hooks are bypassable with --no-verify
and there is no CI to catch it afterwards, so the assertion lives here too --
where the pre-push gate runs the whole suite.

Importing the script is safe: the export happens in ``render()``, not at import
time, so nothing is written by collecting this module.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "openapi.json"


def _load_exporter():
    """scripts/ is not an importable package, so load the module by path."""
    spec = importlib.util.spec_from_file_location(
        "export_openapi", REPO_ROOT / "scripts" / "export_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return _load_exporter()


def test_spec_file_exists():
    assert SPEC_PATH.is_file(), "run: uv run python scripts/export_openapi.py"


def test_spec_file_is_current(exporter):
    """If this fails, re-run scripts/export_openapi.py and commit the result."""
    assert SPEC_PATH.read_text(encoding="utf-8") == exporter.render()


def test_spec_documents_every_route(exporter):
    """Compared against our own router, not ``app.routes``.

    FastAPI wraps an included router in a single opaque entry on the app, so
    ``app.routes`` lists only the /docs and /openapi.json routes it adds
    itself -- an assertion against it passes vacuously.
    """
    from prodllm.api.routes import router

    spec = json.loads(exporter.render())
    declared = {route.path for route in router.routes}
    assert declared == set(spec["paths"])


def test_spec_version_tracks_the_package(exporter):
    import prodllm

    spec = json.loads(exporter.render())
    assert spec["info"]["version"] == prodllm.__version__
