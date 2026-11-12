from __future__ import annotations

import asyncio
import os
from pathlib import Path

from src.agent.tools.base import Prepared, ToolContext, ToolResult
from src.agent.tools.fs import ReadTool


async def _read(ctx: ToolContext, **args: object) -> ToolResult:
    result = await ReadTool().prepare(dict(args), ctx)
    assert isinstance(result, ToolResult)  # read-only: never a Prepared
    assert not isinstance(result, Prepared)
    return result


def _numbered(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(1, n + 1))


async def test_read_returns_numbered_lines(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("one\ntwo\nthree\n")

    result = await _read(ctx, path="a.py")

    assert not result.is_error
    assert result.output == "     1\tone\n     2\ttwo\n     3\tthree"
    assert result.summary == "3 lines"


async def test_read_offset_and_limit_return_lines_10_to_14(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text(_numbered(30))

    result = await _read(ctx, path="a.py", offset=10, limit=5)

    assert result.output.splitlines() == [f"{n:>6}\tline {n}" for n in range(10, 15)]
    assert result.summary == "5 lines"


async def test_read_limit_is_capped_at_2000_lines(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "big.txt").write_text(_numbered(2500))

    result = await _read(ctx, path="big.txt", limit=99999)

    assert result.summary == "2000 lines"
    assert result.output.splitlines()[-1] == f"{2000:>6}\tline 2000"


async def test_read_defaults_to_the_first_2000_lines(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "big.txt").write_text(_numbered(2500))

    result = await _read(ctx, path="big.txt")

    assert result.summary == "2000 lines"


async def test_read_offset_past_the_end_says_so(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("one\n")

    result = await _read(ctx, path="a.py", offset=50)

    assert not result.is_error
    assert result.output == "offset 50 is past the end of the file (1 lines)"
    assert result.summary == "0 lines"


async def test_read_missing_file_is_an_error(ctx: ToolContext) -> None:
    result = await _read(ctx, path="nope.py")

    assert result.is_error
    assert result.output == "No such file: nope.py"


async def test_read_directory_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()

    result = await _read(ctx, path="src")

    assert result.is_error
    assert result.output == "src is a directory"


async def test_read_png_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

    result = await _read(ctx, path="logo.png")

    assert result.is_error
    assert result.output == "logo.png is binary or not UTF-8"


async def test_read_outside_the_workspace_is_allowed(ctx: ToolContext) -> None:
    result = await _read(ctx, path="/etc/hosts")

    assert not result.is_error
    assert result.output


async def test_read_a_file_outside_the_workspace_by_dotdot(workspace: Path, ctx: ToolContext) -> None:
    (workspace.parent / "outside.txt").write_text("outside\n")

    result = await _read(ctx, path="../outside.txt")

    assert result.output == "     1\toutside"


def test_read_label_and_flags() -> None:
    tool = ReadTool()
    assert tool.label({"path": "src/main.py"}) == "read src/main.py"
    assert not tool.mutating
    assert not tool.needs_approval


async def test_read_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _read(ctx, path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


async def test_read_with_an_unknown_tilde_user_reports_no_such_file(ctx: ToolContext) -> None:
    result = await _read(ctx, path="~nouser-xyz/x")

    assert result.is_error
    assert result.output.startswith("No such file")


async def test_read_refuses_a_fifo_without_hanging(workspace: Path, ctx: ToolContext) -> None:
    os.mkfifo(workspace / "pipe")

    result = await asyncio.wait_for(_read(ctx, path="pipe"), 2)

    assert result.is_error
    assert result.output == "pipe is not a regular file"


async def test_read_offset_past_the_end_is_not_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.txt").write_text("one\ntwo\n")

    result = await _read(ctx, path="a.txt", offset=10)

    assert not result.is_error
    assert result.output == "offset 10 is past the end of the file (2 lines)"
    assert result.summary == "0 lines"


async def test_read_offset_just_past_the_end_of_an_empty_file(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "empty.txt").write_text("")

    result = await _read(ctx, path="empty.txt", offset=2)

    assert result.output == "offset 2 is past the end of the file (0 lines)"
