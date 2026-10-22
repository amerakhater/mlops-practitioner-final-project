"""Definition of Done: zero print() statements in src/.

Deliberately redundant with ruff's T20 rule. Lint rules get relaxed; a failing
test gets noticed. This makes the requirement a test result, not a promise.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def iter_source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_src_directory_is_discoverable():
    assert iter_source_files(), f"no Python files found under {SRC}"


def test_no_print_calls_anywhere_in_src():
    offenders: list[str] = []

    for path in iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")

    assert offenders == [], f"print() found in src/: {offenders}"


def test_no_sys_stdout_writes_either():
    """The obvious way to smuggle a print past the rule above."""
    offenders: list[str] = []

    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "sys.stdout.write" in line or "sys.stderr.write" in line:
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")

    assert offenders == [], f"direct stream writes in src/: {offenders}"
