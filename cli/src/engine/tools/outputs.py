"""Putting a tool's output file into the workspace: the path checks and the atomic placement, shared by the image and
media tools."""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from src.engine.guardrails.setup import guard_of

from .base import ToolContext, ToolResult
from .fs import _refuse_outside
from .paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
from .protected import refuse_protected


async def finish_placement(work: Any) -> Any:
    """Wait for the placement thread even when cancelled, then re-raise the cancel. A thread cannot be stopped, so
    once placement started it finishes (file and journal entry together) and /undo covers the file."""
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.done() and not task.cancelled():
                result = task.result()
                cancelled = cancelled or exc
                break
            cancelled = exc
    if cancelled is not None:
        raise cancelled
    return result


class TooBig(Exception):
    pass


def read_nofollow(path: Path, limit: int) -> bytes:
    """A regular file's bytes, never through a symlink, at most `limit` (`TooBig` past it)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise FileNotFoundError(path)
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise TooBig
    return data


def check_output_path(raw: str, ctx: ToolContext, suffix: str = ".png") -> Path | ToolResult:
    """The resolved output path, or the refusal: inside the workspace, not protected, not a folder, ending in `suffix`."""
    try:
        path = resolve_path(ctx, raw)
    except InvalidPath as exc:
        return invalid_path_result(exc)
    if not inside_workspace(path, ctx):
        return _refuse_outside()
    if guard_of(ctx).paths.is_protected(path):
        return refuse_protected()
    if path.is_dir():
        return ToolResult(f"{raw} is a directory", is_error=True, summary="directory")
    if path.exists() and not path.is_file():
        return ToolResult(f"{raw} is not a regular file", is_error=True, summary="not a regular file")
    if path.suffix.lower() != suffix:
        return ToolResult(f"{raw} must end in {suffix}", is_error=True, summary="invalid")
    return path


def _publish(path: Path, ctx: ToolContext, fill: Callable[[IO[bytes]], None]) -> ToolResult | None:
    """Write what `fill` produces beside `path` and rename it into place. A folder may have been swapped for a symlink
    during the minutes a tool ran, so the target is checked again right before the write."""

    def allowed() -> bool:
        final = path.parent.resolve() / path.name
        return inside_workspace(final, ctx) and not guard_of(ctx).paths.is_protected(final)

    if not allowed():
        return _refuse_outside()
    if ctx.journal is not None:
        ctx.journal.before(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not allowed():
            return _refuse_outside()
        try:
            mode = path.stat().st_mode & 0o7777
        except OSError:
            mode = None
        temp = path.with_name(f".{path.name}.agent-{secrets.token_hex(4)}.tmp")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
        try:
            fd = os.open(temp, flags, 0o666)
            with os.fdopen(fd, "wb") as handle:
                fill(handle)
                handle.flush()
                os.fsync(handle.fileno())
                if mode is not None:
                    os.fchmod(handle.fileno(), mode)
            os.replace(temp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                temp.unlink()
            raise
    finally:
        if ctx.journal is not None:
            ctx.journal.after(path)
    return None


def place_bytes(path: Path, data: bytes, ctx: ToolContext) -> ToolResult | None:
    """Place `data` at `path` atomically (see `_publish`); the refusal, or None."""

    def write(handle: IO[bytes]) -> None:
        handle.write(data)

    return _publish(path, ctx, write)


def place_file(path: Path, source: Path, ctx: ToolContext) -> ToolResult | None:
    """Place a copy of the regular file `source` (opened with O_NOFOLLOW) at `path` atomically; the refusal, or None."""

    def copy(handle: IO[bytes]) -> None:
        fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as origin:
            if not stat.S_ISREG(os.fstat(origin.fileno()).st_mode):
                raise FileNotFoundError(source)
            shutil.copyfileobj(origin, handle, 1024 * 1024)

    return _publish(path, ctx, copy)
