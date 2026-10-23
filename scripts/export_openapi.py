"""Write the service's OpenAPI document to openapi.json.

    uv run python scripts/export_openapi.py            # write / refresh it
    uv run python scripts/export_openapi.py --check    # fail if it is stale

FastAPI already serves the same document at ``/openapi.json``, so the point of
a committed copy is everything that happens when the service is *not* running:
reviewing a contract change in a diff, generating a client, importing the API
into a collection, or answering "what does /ask return on 503" without a
container. A spec that only exists at runtime cannot be reviewed before it
ships.

``--check`` is what keeps the file honest. It regenerates the document in
memory and compares; a route or schema edit that was not exported fails the
hook instead of leaving a committed spec that quietly describes the previous
version of the API.

Importing the app is enough -- the lifespan handler never runs, so no weights,
no index and no network are required.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "openapi.json"


def render() -> str:
    """The OpenAPI document, exactly as it should appear on disk."""
    # Imported inside the function so --help does not pay for loading torch.
    from prodllm.api.main import app

    # ensure_ascii=False keeps the Arabic in the schema examples readable in a
    # diff; the trailing newline is what end-of-file-fixer expects.
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write the document (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the file is missing or out of date",
    )
    args = parser.parse_args()

    document = render()

    if args.check:
        if not args.output.is_file():
            print(
                f"ERROR {args.output} does not exist; run without --check",
                file=sys.stderr,
            )
            return 1
        if args.output.read_text(encoding="utf-8") != document:
            print(
                f"ERROR {args.output} is out of date; run without --check",
                file=sys.stderr,
            )
            return 1
        print(f"ok {args.output} matches the application")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly: on Windows, text mode would translate every \n
    # to \r\n, the mixed-line-ending pre-commit hook would convert the file
    # straight back to LF, and the two would take turns rewriting the file on
    # every commit. A generated artifact has the same bytes on every platform.
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)

    spec = json.loads(document)
    paths = spec.get("paths", {})
    operations = sum(len(methods) for methods in paths.values())
    print(f"wrote {args.output}")
    print(f"  openapi version  : {spec.get('openapi')}")
    print(f"  api version      : {spec.get('info', {}).get('version')}")
    print(f"  paths            : {len(paths)} ({operations} operations)")
    print(f"  schemas          : {len(spec.get('components', {}).get('schemas', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
