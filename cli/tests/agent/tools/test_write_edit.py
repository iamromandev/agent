from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.agent.tools.fs import EditTool, WriteTool


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


async def test_edit_previews_a_diff_then_writes_exactly_the_replacement(
    workspace: Path, ctx: ToolContext
) -> None:
    target = workspace / "a.py"
    target.write_text("keep\nold\nkeep too\n")

    prepared = await _prepared(EditTool(), ctx, path="a.py", old_string="old", new_string="new")

    assert prepared.preview is not None
    assert prepared.preview.kind == "diff"
    assert prepared.preview.title == "edit a.py"
    assert "\n-old\n" in prepared.preview.body
    assert "\n+new\n" in prepared.preview.body
    assert target.read_text() == "keep\nold\nkeep too\n"  # prepare has no side effects

    result = await prepared.execute()

    assert not result.is_error
    assert result.output == "replaced 1 occurrence(s) in a.py"
    assert target.read_text() == "keep\nnew\nkeep too\n"


async def test_edit_with_a_missing_old_string_is_an_error_without_a_prepared(
    workspace: Path, ctx: ToolContext
) -> None:
    (workspace / "a.py").write_text("hello\n")

    result = await _refused(EditTool(), ctx, path="a.py", old_string="absent", new_string="x")

    assert result.output == "old_string not found in a.py"


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


async def test_edit_preserves_crlf_line_endings(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")

    prepared = await _prepared(EditTool(), ctx, path="a.txt", old_string="two", new_string="2")
    await prepared.execute()

    assert (workspace / "a.txt").read_bytes() == b"one\r\n2\r\nthree\r\n"


async def test_edit_fails_when_the_file_changed_after_prepare(workspace: Path, ctx: ToolContext) -> None:
    target = workspace / "a.py"
    target.write_text("old\n")
    prepared = await _prepared(EditTool(), ctx, path="a.py", old_string="old", new_string="new")

    target.write_text("changed behind our back\n")
    result = await prepared.execute()

    assert result.is_error
    assert result.output == "a.py changed since the edit was prepared"
    assert target.read_text() == "changed behind our back\n"


async def test_edit_allows_a_symlink_that_stays_inside(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "real.txt").write_text("old\n")
    os.symlink(workspace / "real.txt", workspace / "alias.txt")

    prepared = await _prepared(EditTool(), ctx, path="alias.txt", old_string="old", new_string="new")
    await prepared.execute()

    assert (workspace / "real.txt").read_text() == "new\n"


# --- write (AC13) ----------------------------------------------------------------


async def test_write_creates_missing_folders_after_execute_only(
    workspace: Path, ctx: ToolContext
) -> None:
    prepared = await _prepared(WriteTool(), ctx, path="a/b/c.txt", content="hello\nworld\n")

    assert not (workspace / "a").exists()  # prepare has no side effects
    assert prepared.preview is not None
    assert prepared.preview.kind == "content"
    assert prepared.preview.title == "write a/b/c.txt"
    assert prepared.preview.body == "hello\nworld\n"

    result = await prepared.execute()

    assert not result.is_error
    assert result.output == "wrote 2 lines to a/b/c.txt"
    assert (workspace / "a" / "b" / "c.txt").read_text() == "hello\nworld\n"


async def test_write_over_an_existing_file_previews_a_diff(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.txt").write_text("old\n")

    prepared = await _prepared(WriteTool(), ctx, path="a.txt", content="new\n")

    assert prepared.preview is not None
    assert prepared.preview.kind == "diff"
    assert "\n-old\n" in prepared.preview.body
    assert "\n+new\n" in prepared.preview.body
    assert (workspace / "a.txt").read_text() == "old\n"

    await prepared.execute()

    assert (workspace / "a.txt").read_text() == "new\n"


async def test_write_to_a_directory_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()

    result = await _refused(WriteTool(), ctx, path="src", content="x")

    assert result.output == "src is a directory"


async def test_write_counts_a_last_line_without_a_newline(workspace: Path, ctx: ToolContext) -> None:
    prepared = await _prepared(WriteTool(), ctx, path="a.txt", content="one\ntwo")

    assert (await prepared.execute()).output == "wrote 2 lines to a.txt"


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


async def test_dotdot_path_is_refused_and_the_file_untouched(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside = workspace.parent / "x"
    outside.write_text("secret\n")

    await _assert_refused(mutate, ctx, "../x")

    assert outside.read_text() == "secret\n"


async def test_dotdot_path_to_a_new_outside_file_is_refused_and_not_created(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    await _assert_refused(mutate, ctx, "../created.txt")

    assert not (workspace.parent / "created.txt").exists()


async def test_absolute_outside_path_is_refused_and_the_file_untouched(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside = workspace.parent / "abs.txt"
    outside.write_text("secret\n")

    await _assert_refused(mutate, ctx, str(outside))

    assert outside.read_text() == "secret\n"


async def test_etc_hosts_is_refused(ctx: ToolContext, mutate: Any) -> None:
    before = Path("/etc/hosts").read_bytes()

    await _assert_refused(mutate, ctx, "/etc/hosts")

    assert Path("/etc/hosts").read_bytes() == before


async def test_symlinked_file_pointing_outside_is_refused_and_the_target_untouched(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside = workspace.parent / "target.txt"
    outside.write_text("secret\n")
    os.symlink(outside, workspace / "link.txt")

    await _assert_refused(mutate, ctx, "link.txt")

    assert outside.read_text() == "secret\n"
    assert (workspace / "link.txt").is_symlink()


async def test_dangling_symlink_pointing_outside_is_refused_and_nothing_is_created(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside = workspace.parent / "not-yet.txt"
    os.symlink(outside, workspace / "dangling.txt")

    await _assert_refused(mutate, ctx, "dangling.txt")

    assert not outside.exists()


async def test_symlinked_directory_pointing_outside_is_refused_and_nothing_is_created(
    workspace: Path, ctx: ToolContext, mutate: Any
) -> None:
    outside_dir = workspace.parent / "outdir"
    outside_dir.mkdir()
    (outside_dir / "existing.txt").write_text("secret\n")
    os.symlink(outside_dir, workspace / "linkdir")

    await _assert_refused(mutate, ctx, "linkdir/existing.txt")
    await _assert_refused(mutate, ctx, "linkdir/new/nested.txt")

    assert (outside_dir / "existing.txt").read_text() == "secret\n"
    assert sorted(p.name for p in outside_dir.iterdir()) == ["existing.txt"]


async def test_not_yet_existing_nested_path_inside_is_allowed(workspace: Path, ctx: ToolContext) -> None:
    prepared = await _prepared(WriteTool(), ctx, path="deep/er/new.txt", content="ok\n")

    await prepared.execute()

    assert (workspace / "deep" / "er" / "new.txt").read_text() == "ok\n"


# --- metadata --------------------------------------------------------------------


def test_write_and_edit_flags_and_labels() -> None:
    for tool in (WriteTool(), EditTool()):
        assert tool.mutating
        assert tool.needs_approval
    assert WriteTool().label({"path": "a/b.txt", "content": "x"}) == "write a/b.txt"
    assert EditTool().label({"path": "a.py"}) == "edit a.py"


# --- unresolved workspace (S5) ---------------------------------------------------


async def test_write_works_when_the_workspace_is_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    ctx = ToolContext(workspace=link, bash_timeout=5.0)

    prepared = await _prepared(WriteTool(), ctx, path="a.txt", content="hi\n")
    result = await prepared.execute()

    assert not result.is_error
    assert (real / "a.txt").read_text() == "hi\n"
    refused = await _refused(WriteTool(), ctx, path="../outside.txt", content="x")
    assert refused.summary == "refused: outside workspace"


async def test_write_works_when_the_workspace_is_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "ws").mkdir()
    monkeypatch.chdir(tmp_path)
    ctx = ToolContext(workspace=Path("ws"), bash_timeout=5.0)

    prepared = await _prepared(WriteTool(), ctx, path="a.txt", content="hi\n")
    result = await prepared.execute()

    assert not result.is_error
    assert (tmp_path / "ws" / "a.txt").read_text() == "hi\n"
    refused = await _refused(WriteTool(), ctx, path="../outside.txt", content="x")
    assert refused.summary == "refused: outside workspace"


async def test_edit_works_when_the_workspace_is_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("old\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    ctx = ToolContext(workspace=link, bash_timeout=5.0)

    prepared = await _prepared(EditTool(), ctx, path="a.txt", old_string="old", new_string="new")
    result = await prepared.execute()

    assert not result.is_error
    assert (real / "a.txt").read_text() == "new\n"


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
