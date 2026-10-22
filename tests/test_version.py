"""Version plumbing across the distribution-name / import-name mismatch.

The distribution is `mlops-final-project` but the import package is `prodllm`,
which is exactly the kind of mismatch that makes a version lookup silently
return "0.0.0+unknown" in the container.
"""

import re
from importlib.metadata import version

import prodllm

PEP440 = re.compile(r"^\d+\.\d+\.\d+")


def test_package_exposes_a_version():
    assert PEP440.match(prodllm.__version__), prodllm.__version__


def test_version_matches_the_installed_distribution():
    assert prodllm.__version__ == version("mlops-final-project")


def test_version_is_not_the_unknown_fallback():
    """If this fails, the package is not installed and the image would serve
    a meaningless version from /metadata."""
    assert prodllm.__version__ != "0.0.0+unknown"
