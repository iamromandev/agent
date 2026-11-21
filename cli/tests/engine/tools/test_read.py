from __future__ import annotations

import asyncio
import os
from pathlib import Path

from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.fs import ReadTool


async def _read(ctx: ToolContext, **args: object) -> ToolResult:
    result = await ReadTool().prepare(dict(args), ctx)
    assert isinstance(result, ToolResult)  # read-only: never a Prepared
    assert not isinstance(result, Prepared)
    return result


def _numbered(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(1, n + 1))


async def test_read_offset_past_the_end_says_so(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("one\n")

    result = await _read(ctx, path="a.py", offset=50)

    assert not result.is_error
    assert result.output == "offset 50 is past the end of the file (1 lines)"
    assert result.summary == "0 lines"


async def test_read_directory_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()

    result = await _read(ctx, path="src")

    assert result.is_error
    assert result.output == "src is a directory"


async def test_read_png_without_vision_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

    result = await _read(ctx, path="logo.png")

    assert result.is_error
    assert result.output == "logo.png is an image and the model '' has no vision"
    assert result.images == ()


async def _yes() -> bool:
    return True


async def test_read_png_with_vision_returns_the_image(workspace: Path, ctx: ToolContext) -> None:
    import dataclasses

    data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe"
    (workspace / "logo.png").write_bytes(data)

    result = await _read(dataclasses.replace(ctx, has_vision=_yes, model="v"), path="logo.png", offset=5)

    assert not result.is_error
    assert result.output == "image logo.png (PNG, 1 KB)"
    assert [(i.format, i.data, i.name) for i in result.images] == [("png", data, "logo.png")]


async def test_read_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _read(ctx, path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


async def test_read_refuses_a_fifo_without_hanging(workspace: Path, ctx: ToolContext) -> None:
    os.mkfifo(workspace / "pipe")

    result = await asyncio.wait_for(_read(ctx, path="pipe"), 2)

    assert result.is_error
    assert result.output == "pipe is not a regular file"
