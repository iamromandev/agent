from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from src.engine.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.engine.tools.fs import EditTool, WriteTool


async def _prepare(tool: Tool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    return await tool.prepare(dict(args), ctx)


async def _prepared(tool: Tool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


async def _refused(tool: Tool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


# --- edit (AC13) -----------------------------------------------------------------


async def test_edit_with_a_non_unique_old_string_is_an_error_without_a_prepared(
    workspace: Path, ctx: ToolContext
) -> None:
    (workspace / "a.py").write_text("x\nx\nx\n")

    result = await _refused(EditTool(), ctx, path="a.py", old_string="x", new_string="y")

    assert result.output == "old_string occurs 3 times in a.py; add context or set replace_all"
    assert (workspace / "a.py").read_text() == "x\nx\nx\n"


async def test_edit_replace_all_replaces_every_match(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("x\nx\nx\n")

    prepared = await _prepared(
        EditTool(), ctx, path="a.py", old_string="x", new_string="y", replace_all=True
    )
    result = await prepared.execute()

    assert result.output == "replaced 3 occurrence(s) in a.py"
    assert (workspace / "a.py").read_text() == "y\ny\ny\n"


async def test_edit_of_a_missing_file_is_an_error(ctx: ToolContext) -> None:
    result = await _refused(EditTool(), ctx, path="nope.py", old_string="a", new_string="b")

    assert result.output == "No such file: nope.py"


async def test_edit_of_a_directory_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()

    result = await _refused(EditTool(), ctx, path="src", old_string="a", new_string="b")

    assert result.output == "src is a directory"


async def test_edit_with_an_empty_old_string_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("abc\n")

    result = await _refused(EditTool(), ctx, path="a.py", old_string="", new_string="b")

    assert result.output == "old_string must not be empty"


async def test_edit_of_a_binary_file_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.bin").write_bytes(b"abc\x00def")

    result = await _refused(EditTool(), ctx, path="a.bin", old_string="abc", new_string="b")

    assert result.output == "a.bin is binary or not UTF-8"


async def test_edit_fails_when_the_file_changed_after_prepare(workspace: Path, ctx: ToolContext) -> None:
    target = workspace / "a.py"
    target.write_text("old\n")
    prepared = await _prepared(EditTool(), ctx, path="a.py", old_string="old", new_string="new")

    target.write_text("changed behind our back\n")
    result = await prepared.execute()

    assert result.is_error
    assert result.output == "a.py changed since the edit was prepared"
    assert target.read_text() == "changed behind our back\n"


# --- write (AC13) ----------------------------------------------------------------


async def test_write_to_a_directory_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()

    result = await _refused(WriteTool(), ctx, path="src", content="x")

    assert result.output == "src is a directory"


# --- confinement (AC16, R18, NFR3) -----------------------------------------------


@pytest.fixture(params=["write", "edit"])
def mutate(request: pytest.FixtureRequest) -> Any:
    """Build the same kind of call for either tool; both must be confined alike."""

    def build(path: str) -> tuple[Tool, dict[str, Any]]:
        if request.param == "write":
            return WriteTool(), {"path": path, "content": "PWNED\n"}
        return EditTool(), {"path": path, "old_string": "secret", "new_string": "PWNED"}

    return build


async def _assert_refused(mutate: Any, ctx: ToolContext, path: str) -> None:
    tool, args = mutate(path)
    result = await tool.prepare(args, ctx)
    assert isinstance(result, ToolResult), "refusal must not return a Prepared"
    assert not isinstance(result, Prepared)
    assert result.is_error
    assert result.output == "refused: outside workspace"
    assert result.summary == "refused: outside workspace"


async def test_symlinked_file_pointing_outside_is_refused_and_the_target_untouched(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside = workspace.parent / "target.txt"
    outside.write_text("secret\n")
    os.symlink(outside, workspace / "link.txt")

    await _assert_refused(mutate, ctx, "link.txt")

    assert outside.read_text() == "secret\n"
    assert (workspace / "link.txt").is_symlink()


# --- metadata --------------------------------------------------------------------


# --- unresolved workspace (S5) ---------------------------------------------------


# --- odd model input (S6, N2) ----------------------------------------------------


async def test_write_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _refused(WriteTool(), ctx, path="a\x00b.txt", content="x")

    assert result.output.startswith("Invalid path:")


async def test_edit_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _refused(EditTool(), ctx, path="a\x00b.txt", old_string="a", new_string="b")

    assert result.output.startswith("Invalid path:")


async def test_write_with_a_lone_surrogate_creates_no_directories(workspace: Path, ctx: ToolContext) -> None:
    prepared = await _prepare(WriteTool(), ctx, path="d1/d2/s.txt", content="\ud800")
    if isinstance(prepared, Prepared):
        prepared_result = await prepared.execute()
    else:
        prepared_result = prepared

    assert prepared_result.is_error
    assert prepared_result.output == "content is not valid UTF-8 text"
    assert not (workspace / "d1").exists()


async def test_edit_with_a_lone_surrogate_leaves_the_file_unchanged(workspace: Path, ctx: ToolContext) -> None:
    target = workspace / "a.txt"
    target.write_text("old\n")
    prepared = await _prepare(EditTool(), ctx, path="a.txt", old_string="old", new_string="\ud800")
    result = await prepared.execute() if isinstance(prepared, Prepared) else prepared

    assert result.is_error
    assert result.output == "new_string is not valid UTF-8 text"
    assert target.read_text() == "old\n"


async def test_write_refuses_an_existing_fifo_without_hanging(workspace: Path, ctx: ToolContext) -> None:
    os.mkfifo(workspace / "pipe")

    result = await asyncio.wait_for(_refused(WriteTool(), ctx, path="pipe", content="x"), 2)

    assert result.output == "pipe is not a regular file"


async def test_edit_refuses_an_existing_fifo_without_hanging(workspace: Path, ctx: ToolContext) -> None:
    os.mkfifo(workspace / "pipe")

    result = await asyncio.wait_for(
        _refused(EditTool(), ctx, path="pipe", old_string="a", new_string="b"), 2
    )

    assert result.output == "pipe is not a regular file"


# --- protected paths (guardrails AC7) -------------------------------------------------


async def test_write_refuses_a_protected_path_with_no_change_on_disk(workspace: Path, ctx: ToolContext) -> None:
    result = await _refused(WriteTool(), ctx, path=".env", content="X=1\n")

    assert result.output == "protected path"
    assert result.summary == "refused: protected path"
    assert not (workspace / ".env").exists()


async def test_edit_refuses_a_protected_path_with_no_change_on_disk(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "certs").mkdir()
    pem = workspace / "certs" / "server.pem"
    pem.write_text("old\n")

    result = await _refused(EditTool(), ctx, path="certs/server.pem", old_string="old", new_string="new")

    assert result.output == "protected path"
    assert pem.read_text() == "old\n"


# --- the redaction placeholder is never written ---------------------------------------------------------------------


async def test_edit_refuses_a_new_string_with_a_redaction_placeholder(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "cfg.txt").write_text("token=real\n")

    result = await _refused(EditTool(), ctx, path="cfg.txt", old_string="real", new_string="[redacted:env:API_TOKEN]")

    assert "placeholder" in result.output
    assert (workspace / "cfg.txt").read_text() == "token=real\n"


async def test_write_refuses_a_placeholder_the_file_does_not_have_yet(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "doc.md").write_text("A result shows [redacted:github-token].\n")

    result = await _refused(
        WriteTool(), ctx, path="doc.md", content="A result shows [redacted:github-token] and [redacted:jwt].\n"
    )

    assert "placeholder" in result.output
    assert (workspace / "doc.md").read_text() == "A result shows [redacted:github-token].\n"


@pytest.mark.parametrize("text", ["[redacted:env:]"])
async def test_text_that_only_looks_a_little_like_a_placeholder_is_written(workspace: Path, ctx: ToolContext, text: str) -> None:
    prepared = await _prepared(WriteTool(), ctx, path="note.txt", content=f"{text}\n")

    assert not (await prepared.execute()).is_error
