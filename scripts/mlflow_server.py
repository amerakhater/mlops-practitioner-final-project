"""Start the MLflow tracking server locally, without Docker.

    uv run --group tracking python scripts/mlflow_server.py

Equivalent to the `mlflow` service in docker-compose.yml, for when the stack
is not running. The UI is then at http://127.0.0.1:5000 and the sweep picks it
up through its --tracking-uri default.

The one thing this wraps that is worth wrapping is the backend store URI.
SQLite needs an absolute path with the right number of slashes, and on Windows
a hand-written `sqlite:///./mlruns/mlflow.db` silently produces a database in a
different directory than the one the next invocation reads. Getting that wrong
looks exactly like "my runs disappeared".
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_STORE = Path("mlruns")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help="Directory holding mlflow.db and the artifact root.",
    )
    args = parser.parse_args()

    store = args.store.resolve()
    artifacts = store / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # A database-backed store, not a file store: the Model Registry is only
    # available with one, and registering the winning configuration is half of
    # what this module is for.
    backend = f"sqlite:///{store.as_posix()}/mlflow.db"

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--backend-store-uri",
        backend,
        "--artifacts-destination",
        str(artifacts),
        "--serve-artifacts",
    ]
    print(f"backend store : {backend}")
    print(f"artifact root : {artifacts}")
    print(f"UI            : http://{args.host}:{args.port}")
    print("Ctrl-C to stop.\n")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
