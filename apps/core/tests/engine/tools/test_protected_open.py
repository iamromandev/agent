"""A symlink changed between the path check and the open must not get a protected file past the read tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from core.engine.guardrails.setup import prepare
from core.engine.tools import fs
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.fs import ReadTool
from core.engine.tools.search import GrepTool


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    (folder / ".aws").mkdir(parents=True)
    (folder / ".aws" / "credentials").write_text("aws_secret=hunter2\n")
    return folder.resolve()


@pytest.fixture
def gctx(workspace: Path, home: Path) -> ToolContext:
    return ToolContext(workspace, 5.0, guard=prepare(workspace, None, None, {}, home=home))


@pytest.fixture
def link(workspace: Path) -> Path:
    """`link.txt` points at a harmless file now; the tests move it to a secret just before the file is opened."""
    (workspace / "harmless.txt").write_text("aws_secret=nothing\n")
    link = workspace / "link.txt"
    link.symlink_to(workspace / "harmless.txt")
    return link


@pytest.fixture
def swap_on_open(monkeypatch: pytest.MonkeyPatch, link: Path) -> Callable[[Path], None]:
    """`swap_on_open(secret)`: the first `os.open` of the link finds it pointing at `secret`."""

    def arm(secret: Path) -> None:
        real_open = os.open
        done = []

        def hooked(path: str | os.PathLike[str], *args: int, **kwargs: int) -> int:
            if not done and Path(path) == link:
                done.append(True)
                link.unlink()
                link.symlink_to(secret)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", hooked)
        # `resolve_path` would resolve the link to the harmless file at check time and the open would never see it
        monkeypatch.setattr(fs, "resolve_path", lambda ctx, raw: link)
        monkeypatch.setattr("core.engine.tools.search.resolve_path", lambda ctx, raw: link)

    return arm


@pytest.fixture(params=["named", "in_folder"])
def secret(request: pytest.FixtureRequest, workspace: Path, home: Path) -> Path:
    if request.param == "named":
        target = workspace.parent / "outside" / "prod.pem"  # outside, so the folder listing does not skip it itself
        target.parent.mkdir()
        target.write_text("aws_secret=hunter2\n")
        return target
    return home / ".aws" / "credentials"


async def _read(ctx: ToolContext) -> ToolResult:
    result = await ReadTool().prepare({"path": "link.txt"}, ctx)
    assert not isinstance(result, Prepared)
    return result


async def test_a_read_of_a_hard_link_to_a_protected_file_is_refused_by_the_open_file(
    gctx: ToolContext, workspace: Path, home: Path
) -> None:
    os.link(home / ".aws" / "credentials", workspace / "copy.txt")  # made after the guard was built, still the file

    result = await ReadTool().prepare({"path": "copy.txt"}, gctx)

    assert isinstance(result, ToolResult)
    assert result.output == "protected path"


async def test_a_targeted_grep_refuses_a_link_that_became_a_protected_file_after_the_check(
    gctx: ToolContext, swap_on_open: Callable[[Path], None], secret: Path
) -> None:
    swap_on_open(secret)

    result = await GrepTool().prepare({"pattern": "aws_secret", "path": "link.txt"}, gctx)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.output == "protected path"


async def test_a_grep_over_a_folder_skips_a_link_that_became_protected_after_the_listing_check(
    gctx: ToolContext, workspace: Path, link: Path, monkeypatch: pytest.MonkeyPatch, secret: Path
) -> None:
    real_open = os.open
    done = []

    def hooked(path: str | os.PathLike[str], *args: int, **kwargs: int) -> int:
        if not done and Path(path) == link:
            done.append(True)
            link.unlink()
            link.symlink_to(secret)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", hooked)

    result = await GrepTool().prepare({"pattern": "aws_secret"}, gctx)

    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert "hunter2" not in result.output
    assert result.output.splitlines()[-1] == "1 protected files skipped"
    assert done


async def test_a_fifo_swapped_in_after_the_check_is_not_opened_blocking(
    gctx: ToolContext, workspace: Path, link: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = workspace / "pipe"
    os.mkfifo(fifo)
    real_open = os.open

    def hooked(path: str | os.PathLike[str], *args: int, **kwargs: int) -> int:
        if Path(path) == link:
            link.unlink()
            link.symlink_to(fifo)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", hooked)
    monkeypatch.setattr(fs, "resolve_path", lambda ctx, raw: link)

    result = await _read(gctx)

    assert result.is_error
    assert "not a regular file" in result.output
