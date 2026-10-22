"""Arabic legal document question answering over the Egyptian Civil Code.

Retrieval-augmented: article-level chunks are embedded and searched, and every
answer carries the article citations it was drawn from.

The import package is ``prodllm``; the installed distribution is named
``mlops-final-project``, which is why the version lookup below uses the
distribution name rather than ``__name__``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlops-final-project")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
