"""write and edit tell the journal before and after they change a file."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from core.engine.tools.base import Prepared, ToolContext
from core.engine.tools.fs import EditTool, WriteTool


class RecordingJournal:
    def __init__(self, workspace: Path) -> None:
        self.calls: list[tuple[str, Path, bool]] = []  # (kind, path, file exists at that moment)
        self.workspace = workspace

    def before(self, path: Path) -> None:
        self.calls.append(("before", path, path.exists()))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path, path.exists()))


@pytest.fixture
def journal(workspace: Path) -> RecordingJournal:
    return RecordingJournal(workspace)


@pytest.fixture
def jctx(ctx: ToolContext, journal: RecordingJournal) -> ToolContext:
    return dataclasses.replace(ctx, journal=journal)


async def prepared(tool: WriteTool | EditTool, ctx: ToolContext, **args: object) -> Prepared:
    result = await tool.prepare(dict(args), ctx)
    assert isinstance(result, Prepared), result
    return result


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
async def test_a_write_that_fails_still_calls_after(
    workspace: Path, jctx: ToolContext, journal: RecordingJournal
) -> None:
    folder = workspace / "ro"
    folder.mkdir()
    run = await prepared(WriteTool(), jctx, path="ro/n.txt", content="x")
    folder.chmod(0o500)
    try:
        result = await run.execute()
    finally:
        folder.chmod(0o700)

    assert result.is_error
    target = folder / "n.txt"
    assert journal.calls == [("before", target, False), ("after", target, False)]
