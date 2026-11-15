"""Reading a settings file that may have been written by someone else (a cloned repository)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

MAX_FILE_BYTES = 1 << 20  # a settings file is a few lines: more than 1 MiB is not one


def read_regular(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    """The bytes of the regular file at `path`.

    Raises `FileNotFoundError` when it is missing and `OSError` when it is anything but a regular file (a FIFO
    would block the read forever, a device never ends) or is larger than `limit`. The type is checked on the open
    file, so nothing can swap in a FIFO between the check and the read.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)  # NONBLOCK: opening a FIFO must not wait for a writer
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        if info.st_size > limit:
            raise OSError(f"larger than {limit} bytes")
        data = handle.read(limit + 1)
    if len(data) > limit:  # it grew while it was read
        raise OSError(f"larger than {limit} bytes")
    return data


def parse_json(text: str) -> object:
    """`json.loads`, where every way to fail is a `ValueError` (deep nesting is a `RecursionError` there)."""
    try:
        return json.loads(text)
    except (RecursionError, MemoryError):
        raise ValueError("nested too deeply or too large") from None
