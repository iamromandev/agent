"""When a file or folder was made: `st_birthtime` where `os.stat` has it (macOS), else `statx` (Linux).

`ctime` is no substitute: it moves on every write, rename or chmod, and a folder's moves whenever an entry is added
or removed, so an old folder the turn wrote into would look new. Python (3.15) has no `os.statx`, so the glibc/musl
wrapper is called through `ctypes`; a system or file system without birth times gives None."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable

AT_SYMLINK_NOFOLLOW = 0x100
STATX_BTIME = 0x800
STATX_SIZE = 256  # sizeof(struct statx)
STX_INO = 32  # offset of stx_ino (u64)
STX_BTIME = 80  # offset of stx_btime: struct statx_timestamp {s64 tv_sec; u32 tv_nsec; s32 reserved}
SETTLE_LIMIT = 0.05  # seconds; a tick is 1-10 ms
# CLOCK_REALTIME_COARSE (<linux/time.h>): the clock Linux stamps files with. Python does not export the name.
_COARSE: int | None = 5 if sys.platform.startswith("linux") else None

StatxFn = Callable[[int, bytes, int, int, ctypes.Array[ctypes.c_char]], int]


def _load_statx() -> StatxFn | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        fn = ctypes.CDLL(None, use_errno=True).statx
    except (OSError, AttributeError):  # no libc symbol (glibc before 2.28, old musl)
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


_statx: StatxFn | None = _load_statx()


def birth_time(dir_fd: int, name: str, info: os.stat_result) -> float | None:
    """The birth time of `name` in the folder `dir_fd` (not following a symlink), whose `lstat` is `info`; None when
    the system keeps none or the entry is no longer the one `info` describes. Never raises."""
    born = getattr(info, "st_birthtime", None)
    if born is not None:
        return float(born)
    if _statx is None:
        return None
    buf = ctypes.create_string_buffer(STATX_SIZE)
    try:
        if _statx(dir_fd, os.fsencode(name), AT_SYMLINK_NOFOLLOW, STATX_BTIME, buf) != 0:
            return None
    except (OSError, ValueError):
        return None
    raw = buf.raw
    if not int.from_bytes(raw[0:4], sys.byteorder) & STATX_BTIME:
        return None  # the file system keeps no birth time
    if int.from_bytes(raw[STX_INO : STX_INO + 8], sys.byteorder) != info.st_ino:
        return None  # swapped since the lstat: the caller falls back to `info`
    seconds = int.from_bytes(raw[STX_BTIME : STX_BTIME + 8], sys.byteorder, signed=True)
    nanos = int.from_bytes(raw[STX_BTIME + 8 : STX_BTIME + 12], sys.byteorder)
    return seconds + nanos / 1e9


def settle(mark: float) -> None:
    """Wait until a file made from now on is stamped after `mark` (a `time.time()`). Linux stamps files from a coarse
    clock that runs up to one tick (1-10 ms) behind, so a file made right after `mark` could look older than it.
    Returns at once where there is no coarse clock, and after `SETTLE_LIMIT` at most."""
    if _COARSE is None:
        return
    deadline = time.monotonic() + SETTLE_LIMIT
    try:
        while time.clock_gettime(_COARSE) <= mark and time.monotonic() < deadline:
            time.sleep(0.001)
    except OSError:  # a kernel without the clock
        return
