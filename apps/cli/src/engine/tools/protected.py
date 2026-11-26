"""Opening a file for a read tool without trusting an earlier path check (a symlink can change in between)."""

from __future__ import annotations

import io
import os
import stat
from pathlib import Path
from typing import BinaryIO

from src.engine.guardrails.setup import guard_of

from .base import ToolContext, ToolResult
from .text import SNIFF_BYTES, head_is_binary

REFUSED_PROTECTED = "refused: protected path"


class ProtectedFile(Exception):
    """The file that was opened is a protected one, and the user did not allow this call to read it."""


class NotRegular(OSError):
    """What was opened is not a regular file (a FIFO or device swapped in after the check)."""


def refuse_protected() -> ToolResult:
    return ToolResult("protected path", is_error=True, summary=REFUSED_PROTECTED)


def open_checked(path: Path, ctx: ToolContext) -> BinaryIO:
    """Open `path` for reading and look at the open file itself: a regular file, and, unless the loop recorded the
    user's say-so for this call (`ctx.protected_ok`), not a protected one. Raises `ProtectedFile`, `NotRegular` or
    `OSError`."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)  # NONBLOCK: a FIFO swapped in must not hang the open
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NotRegular("not a regular file")
        if not ctx.protected_ok and guard_of(ctx).paths.is_protected_fd(fd):
            raise ProtectedFile(str(path))
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def open_text_checked(path: Path, ctx: ToolContext) -> tuple[io.TextIOWrapper, bool]:
    """`open_checked` as UTF-8 text from the first byte, and whether the first 8 KB look binary."""
    return as_text(open_checked(path, ctx))


def as_text(raw: BinaryIO) -> tuple[io.TextIOWrapper, bool]:
    """An already checked open file as UTF-8 text from the first byte, and whether its first 8 KB look binary."""
    try:
        binary = head_is_binary(raw.read(SNIFF_BYTES))
        raw.seek(0)
        return io.TextIOWrapper(raw, encoding="utf-8"), binary
    except BaseException:
        raw.close()
        raise
