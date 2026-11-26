"""Unified lint + type-check runner.

Usage:

    uv run python -m scripts.lint           # check only
    uv run python -m scripts.lint --fix     # auto-fix lint where possible

Exit code is the number of failed steps (0 = all passed).

``ruff format`` is deliberately absent: since 0.15 it rewrites valid
``except (A, B):`` into invalid Python.
"""

from __future__ import annotations

import subprocess
import sys
from typing import NoReturn


def _run(cmd: list[str], description: str) -> bool:
    print(f"\n── {description} ──", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"✗ {description} FAILED (exit={result.returncode})", flush=True)
        return False
    print(f"✓ {description} passed", flush=True)
    return True


def main() -> NoReturn:
    fix = "--fix" in sys.argv

    steps: list[tuple[list[str], str]] = [
        (["uv", "run", "ruff", "check", "--fix", "."] if fix else ["uv", "run", "ruff", "check", "."], "Lint"),
        (["uv", "run", "ty", "check"], "Type check"),
    ]

    failures = [desc for cmd, desc in steps if not _run(cmd, desc)]
    total = len(steps) - len(failures)

    print(f"\n{'=' * 40}")
    print(f"Results: {total}/{len(steps)} passed")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        sys.exit(len(failures))
    sys.exit(0)


if __name__ == "__main__":
    main()
