from __future__ import annotations

import time
from pathlib import Path

from src.agent.tools.base import ToolContext, ToolResult
from src.agent.tools.search import GlobTool, GrepTool

LIMIT_SECONDS = 3.0  # NFR6 / AC23


def _generate(root: Path) -> None:
    for d in range(100):
        folder = root / f"dir{d:03}"
        folder.mkdir()
        for f in range(100):
            (folder / f"file{f:03}.txt").write_text(f"line one\nvalue {d}-{f}\nneedle_{d}_{f}\n")


async def test_glob_and_grep_over_10000_files_finish_under_3_seconds(workspace: Path) -> None:
    _generate(workspace)
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0)

    start = time.perf_counter()
    glob = await GlobTool().prepare({"pattern": "**/*.txt"}, ctx)
    glob_seconds = time.perf_counter() - start

    start = time.perf_counter()
    grep = await GrepTool().prepare({"pattern": r"needle_\d+_99$"}, ctx)
    grep_seconds = time.perf_counter() - start

    print(f"AC23 glob={glob_seconds:.3f}s grep={grep_seconds:.3f}s")
    assert isinstance(glob, ToolResult)
    assert isinstance(grep, ToolResult)
    assert glob.output.splitlines()[-1] == "... 9800 more"
    assert grep.summary == "100 matches"
    assert glob_seconds < LIMIT_SECONDS
    assert grep_seconds < LIMIT_SECONDS
