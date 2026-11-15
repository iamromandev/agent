"""Workspace path resolution and confinement checks."""

from __future__ import annotations

from pathlib import Path

from .base import ToolContext, ToolResult


class InvalidPath(ValueError):
    """The model gave a path the OS cannot represent (for example a NUL byte)."""


def invalid_path_result(exc: InvalidPath) -> ToolResult:
    return ToolResult(f"Invalid path: {exc}", is_error=True, summary="invalid path")


def resolve_path(ctx: ToolContext, raw: str) -> Path:
    """Resolve `raw` against the workspace, following symlinks.

    `Path.resolve()` (non-strict) resolves every existing parent, including symlinked
    files and directories, and appends the not-yet-existing tail unchanged.
    A `~user` that cannot be expanded is kept as a literal relative path.
    Raises InvalidPath for a path the OS rejects.
    """
    try:
        path = Path(raw).expanduser()
    except RuntimeError:  # "~nouser/x": no such home directory
        path = Path(raw)
    try:
        if not path.is_absolute():
            path = ctx.workspace.resolve() / path
        return path.resolve()
    except (ValueError, RuntimeError, OSError) as exc:
        raise InvalidPath(f"{raw!r}: {exc}") from exc


def inside_workspace(path: Path, ctx: ToolContext) -> bool:
    return path.is_relative_to(ctx.workspace.resolve())
