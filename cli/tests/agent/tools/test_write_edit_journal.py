"""write and edit tell the journal before and after they change a file."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from src.agent.tools.base import Prepared, ToolContext
from src.agent.tools.fs import EditTool, WriteTool


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


async def test_write_to_a_new_file_calls_before_while_it_is_missing_and_after_once_written(
    workspace: Path, jctx: ToolContext, journal: RecordingJournal
) -> None:
    run = await prepared(WriteTool(), jctx, path="sub/n.txt", content="x")
    assert journal.calls == []  # prepare has no side effects

    await run.execute()

    target = workspace / "sub" / "n.txt"
    assert journal.calls == [("before", target, False), ("after", target, True)]


async def test_write_over_an_existing_file_calls_before_while_the_old_content_is_there(
    workspace: Path, jctx: ToolContext, journal: RecordingJournal
) -> None:
    target = workspace / "a.txt"
    target.write_text("old")
    run = await prepared(WriteTool(), jctx, path="a.txt", content="new")

    await run.execute()

    assert journal.calls == [("before", target, True), ("after", target, True)]


async def test_edit_calls_before_and_after_around_the_write(
    workspace: Path, jctx: ToolContext, journal: RecordingJournal
) -> None:
    target = workspace / "a.txt"
    target.write_text("old")
    run = await prepared(EditTool(), jctx, path="a.txt", old_string="old", new_string="new")

    await run.execute()

    assert journal.calls == [("before", target, True), ("after", target, True)]
    assert target.read_text() == "new"


async def test_edit_refused_because_the_file_changed_makes_no_journal_call(
    workspace: Path, jctx: ToolContext, journal: RecordingJournal
) -> None:
    target = workspace / "a.txt"
    target.write_text("old")
    run = await prepared(EditTool(), jctx, path="a.txt", old_string="old", new_string="new")
    target.write_text("changed meanwhile")

    result = await run.execute()

    assert result.is_error
    assert journal.calls == []


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


async def test_without_a_journal_write_and_edit_work_as_before(workspace: Path, ctx: ToolContext) -> None:
    assert ctx.journal is None
    run = await prepared(WriteTool(), ctx, path="a.txt", content="x")

    result = await run.execute()

    assert not result.is_error
    assert (workspace / "a.txt").read_text() == "x"
