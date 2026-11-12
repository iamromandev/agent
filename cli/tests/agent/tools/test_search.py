from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from src.agent.tools.base import Prepared, ToolContext, ToolResult
from src.agent.tools.search import GlobTool, GrepTool


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


async def test_glob_lists_matches_and_skips_git_and_ignored(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="**/*.py")

    assert not result.is_error
    assert result.output.splitlines() == ["src/main.py"]
    assert result.summary == "1 files"


async def test_glob_with_a_path_is_relative_to_the_workspace(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*.py", path="src")

    assert result.output.splitlines() == ["src/main.py"]


async def test_glob_outside_the_workspace_returns_absolute_paths(
    workspace: Path, ctx: ToolContext
) -> None:
    outside = workspace.parent / "elsewhere"
    outside.mkdir()
    (outside / "x.py").write_text("x\n")

    result = await _run(GlobTool(), ctx, pattern="*.py", path=str(outside))

    assert result.output.splitlines() == [str(outside / "x.py")]


async def test_glob_caps_at_200_and_counts_the_rest(workspace: Path, ctx: ToolContext) -> None:
    for i in range(250):
        (workspace / f"f{i:03}.txt").write_text("x")

    result = await _run(GlobTool(), ctx, pattern="*.txt")

    lines = result.output.splitlines()
    assert len(lines) == 201
    assert lines[-1] == "... 50 more"
    assert lines[0] == "f000.txt"


async def test_glob_with_no_match_says_so(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="**/*.rs")

    assert not result.is_error
    assert result.output == "No files matched"


async def test_glob_missing_directory_is_an_error(ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*", path="nope")

    assert result.is_error
    assert result.output == "No such directory: nope"


async def test_grep_finds_a_match_with_file_and_line_number(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="def main", include="*.py")

    assert not result.is_error
    assert result.output == "src/main.py:3: def main() -> None:"
    assert result.summary == "1 matches"


async def test_grep_skips_git_dir_and_ignored_files(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="def main")

    assert sorted(result.output.splitlines()) == [
        "src/main.py:3: def main() -> None:",
        "src/util.txt:1: def main is mentioned here",
    ]


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


async def test_grep_single_file_path_searches_only_that_file(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="def main", path="src/util.txt")

    assert result.output == "src/util.txt:1: def main is mentioned here"


async def test_grep_skips_binary_files(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "blob.bin").write_bytes(b"needle\x00\x01\x02")
    (workspace / "text.txt").write_text("needle\n")

    result = await _run(GrepTool(), ctx, pattern="needle")

    assert result.output == "text.txt:1: needle"


async def test_grep_strips_trailing_whitespace_only(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.txt").write_text("    indented   \n")

    result = await _run(GrepTool(), ctx, pattern="indented")

    assert result.output == "a.txt:1:     indented"


async def test_grep_include_matches_nested_file_names(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "a.py").write_text("hit\n")
    (workspace / "pkg" / "a.txt").write_text("hit\n")

    result = await _run(GrepTool(), ctx, pattern="hit", include="*.py")

    assert result.output == "pkg/a.py:1: hit"


async def test_grep_with_no_match_says_so(tree: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="zzzzzz")

    assert not result.is_error
    assert result.output == "No matches"


async def test_grep_missing_path_is_an_error(ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="x", path="nope")

    assert result.is_error
    assert result.output == "No such file or directory: nope"


def test_labels_and_flags() -> None:
    assert GlobTool().label({"pattern": "**/*.py"}) == "glob **/*.py"
    assert GrepTool().label({"pattern": "def main"}) == "grep def main"
    for tool in (GlobTool(), GrepTool()):
        assert not tool.mutating
        assert not tool.needs_approval


async def test_glob_shows_workspace_relative_paths_for_a_symlinked_workspace(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "src").mkdir(parents=True)
    (real / "src" / "a.py").write_text("x\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    ctx = ToolContext(workspace=link, bash_timeout=5.0)

    result = await _run(GlobTool(), ctx, pattern="**/*.py")

    assert result.output == "src/a.py"


async def test_glob_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="*", path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


async def test_grep_with_a_nul_byte_in_the_path_is_an_error_result(ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="x", path="a\x00b")

    assert result.is_error
    assert result.output.startswith("Invalid path:")


def test_glob_and_grep_descriptions_say_brace_expansion_is_not_supported() -> None:
    for tool in (GlobTool(), GrepTool()):
        assert "{py,js}" in tool.description
        assert "once per extension" in tool.description
    include = GrepTool().parameters["properties"]["include"]["description"]
    assert "{py,js}" in include


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


async def test_glob_includes_files_of_an_untracked_nested_repo(nested_repo: Path, ctx: ToolContext) -> None:
    result = await _run(GlobTool(), ctx, pattern="**/*.py")

    assert result.output.splitlines() == ["nested/in.py", "top.py"]


async def test_grep_searches_an_untracked_nested_repo(nested_repo: Path, ctx: ToolContext) -> None:
    result = await _run(GrepTool(), ctx, pattern="inner")

    assert result.output == "nested/in.py:1: inner"
