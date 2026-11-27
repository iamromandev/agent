from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from core.engine.guardrails.setup import Guardrails, prepare
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.search import GlobTool, GrepTool


def prepare_guard(home: Path) -> Guardrails:
    return prepare(home / "ws", None, None, {}, home=home)


async def _run(tool: GlobTool | GrepTool, ctx: ToolContext, **args: object) -> ToolResult:
    result = await tool.prepare(dict(args), ctx)
    assert not isinstance(result, Prepared)
    return result


@pytest.fixture
def tree(workspace: Path) -> Path:
    """src/main.py, a .git/ folder and a .gitignore listing build/ (real git repo)."""
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("import os\n\ndef main() -> None:\n    pass\n")
    (workspace / "src" / "util.txt").write_text("def main is mentioned here\n")
    (workspace / "build").mkdir()
    (workspace / "build" / "gen.py").write_text("def main() -> None: ...\n")
    (workspace / ".gitignore").write_text("build/\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / ".git" / "hooks" / "hook.py").write_text("def main(): ...\n")
    return workspace


async def test_glob_outside_the_workspace_returns_absolute_paths(
    workspace: Path, ctx: ToolContext
) -> None:
    outside = workspace.parent / "elsewhere"
    outside.mkdir()
    (outside / "x.py").write_text("x\n")

    result = await _run(GlobTool(), ctx, pattern="*.py", path=str(outside))

    assert result.output.splitlines() == [str(outside / "x.py")]


async def test_glob_with_no_match_says_so(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="**/*.rs")

    assert not result.is_error
    assert result.output == "No files matched"


async def test_glob_missing_directory_is_an_error(ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*", path="nope")

    assert result.is_error
    assert result.output == "No such directory: nope"


async def test_grep_caps_at_200_and_counts_the_rest(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "many.txt").write_text("".join(f"match {i}\n" for i in range(250)))

    result = await _run(GrepTool(), ctx, pattern="match")

    lines = result.output.splitlines()
    assert len(lines) == 201
    assert lines[0] == "many.txt:1: match 0"
    assert lines[199] == "many.txt:200: match 199"
    assert lines[-1] == "... 50 more"


async def test_grep_invalid_regex_is_an_error(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="(")

    assert result.is_error
    assert result.output.startswith("Invalid regex: ")


async def test_grep_skips_binary_files(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "blob.bin").write_bytes(b"needle\x00\x01\x02")
    (workspace / "text.txt").write_text("needle\n")

    result = await _run(GrepTool(), ctx, pattern="needle")

    assert result.output == "text.txt:1: needle"


async def test_grep_missing_path_is_an_error(ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="x", path="nope")

    assert result.is_error
    assert result.output == "No such file or directory: nope"


async def test_glob_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*", path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


async def test_grep_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="x", path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


@pytest.fixture
def nested_repo(workspace: Path) -> Path:
    (workspace / "top.py").write_text("top\n")
    _git_init(workspace)
    nested = workspace / "nested"
    nested.mkdir()
    _git_init(nested)
    (nested / "in.py").write_text("inner\n")
    return workspace


# --- protected paths (guardrails AC6) -------------------------------------------------


@pytest.fixture
def secrets(workspace: Path) -> Path:
    (workspace / ".env").write_text("password=hunter2\n")
    (workspace / "app.py").write_text("password = os.environ['P']\n")
    (workspace / "server.pem").write_text("password\n")
    (workspace / ".env.example").write_text("password=\n")
    return workspace


async def test_grep_with_nothing_else_to_find_still_reports_the_skipped_files(secrets: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="hunter2")

    assert result.output == "No matches\n2 protected files skipped"


async def test_grep_counts_only_protected_files_its_include_filter_would_search(secrets: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="password", include="*.py")

    assert result.output == "app.py:1: password = os.environ['P']"


async def test_glob_leaves_out_protected_files_and_says_how_many(secrets: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*")

    assert result.output.splitlines() == [".env.example", "app.py", "2 protected files skipped"]


async def test_files_under_a_symlinked_folder_pointing_at_a_protected_one_are_skipped(
    workspace: Path, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host x\n")
    (workspace / "linked").symlink_to(home / ".ssh")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)  # git lists the link; a plain walk would not enter it
    guarded = ToolContext(workspace, 5.0, guard=prepare_guard(home))

    result = await _run(GrepTool(), guarded, pattern="Host")

    assert result.output == "No matches\n1 protected files skipped"
