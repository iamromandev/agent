from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.agent.tools.base import ToolContext
from src.agent.tools.paths import InvalidPath, inside_workspace, resolve_path


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(workspace=workspace.resolve(), bash_timeout=5.0)


def test_relative_path_resolves_inside_the_workspace(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "src/a.py")
    assert path == ctx.workspace / "src" / "a.py"
    assert inside_workspace(path, ctx)


def test_dotdot_escaping_the_workspace_is_outside(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "../x")
    assert path == ctx.workspace.parent / "x"
    assert not inside_workspace(path, ctx)


def test_dotdot_staying_inside_is_inside(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "a/../b.txt")
    assert path == ctx.workspace / "b.txt"
    assert inside_workspace(path, ctx)


def test_absolute_outside_path_is_outside(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "/etc/hosts")
    assert not inside_workspace(path, ctx)


def test_absolute_inside_path_is_inside(ctx: ToolContext) -> None:
    path = resolve_path(ctx, str(ctx.workspace / "a.txt"))
    assert inside_workspace(path, ctx)


def test_tilde_is_expanded(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    path = resolve_path(ctx, "~/notes.txt")
    assert path == home.resolve() / "notes.txt"
    assert not inside_workspace(path, ctx)


def test_symlinked_file_pointing_outside_is_outside(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    os.symlink(outside, ctx.workspace / "link.txt")
    path = resolve_path(ctx, "link.txt")
    assert path == outside.resolve()
    assert not inside_workspace(path, ctx)


def test_dangling_symlink_pointing_outside_is_outside(ctx: ToolContext, tmp_path: Path) -> None:
    os.symlink(tmp_path / "not-there.txt", ctx.workspace / "dangling.txt")
    assert not inside_workspace(resolve_path(ctx, "dangling.txt"), ctx)


def test_symlinked_directory_pointing_outside_is_outside(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "outdir"
    outside.mkdir()
    os.symlink(outside, ctx.workspace / "linkdir")
    path = resolve_path(ctx, "linkdir/new/file.txt")  # nothing below linkdir exists yet
    assert path == outside.resolve() / "new" / "file.txt"
    assert not inside_workspace(path, ctx)


def test_symlink_inside_the_workspace_stays_inside(ctx: ToolContext) -> None:
    (ctx.workspace / "real.txt").write_text("x")
    os.symlink(ctx.workspace / "real.txt", ctx.workspace / "alias.txt")
    assert inside_workspace(resolve_path(ctx, "alias.txt"), ctx)


def test_not_yet_existing_nested_path_inside_is_inside(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "a/b/c/new.txt")
    assert path == ctx.workspace / "a" / "b" / "c" / "new.txt"
    assert inside_workspace(path, ctx)


def test_sibling_directory_sharing_the_prefix_is_outside(ctx: ToolContext) -> None:
    sibling = ctx.workspace.parent / (ctx.workspace.name + "-evil")
    assert not inside_workspace(sibling / "x", ctx)


def test_a_nul_byte_in_the_path_raises_invalid_path(ctx: ToolContext) -> None:
    with pytest.raises(InvalidPath):
        resolve_path(ctx, "a\x00b.txt")


def test_invalid_path_is_a_value_error() -> None:
    assert issubclass(InvalidPath, ValueError)


def test_tilde_with_an_unknown_user_is_a_literal_relative_path(ctx: ToolContext) -> None:
    path = resolve_path(ctx, "~nouser-xyz/x")
    assert path == ctx.workspace / "~nouser-xyz" / "x"
    assert inside_workspace(path, ctx)


def test_a_symlinked_workspace_is_compared_resolved(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    ctx = ToolContext(workspace=link, bash_timeout=5.0)
    assert inside_workspace(resolve_path(ctx, "a.txt"), ctx)
    assert not inside_workspace(resolve_path(ctx, "../a.txt"), ctx)
