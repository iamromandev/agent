"""`python -m core.server serve --workspace DIR`: the daemon process `start_background` starts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.server")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("serve", help="run the agent daemon for a workspace in the foreground")
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--model", default=None, help="the model a newly started daemon runs")
    args = parser.parse_args(argv)
    from core.server.serve import serve

    return serve(args.workspace, model=args.model)


if __name__ == "__main__":
    sys.exit(main())
