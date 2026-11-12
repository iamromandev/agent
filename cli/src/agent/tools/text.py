"""Small text helpers shared by the tools and the loop."""

from __future__ import annotations

import codecs
import difflib
import re
from pathlib import Path

TOOL_OUTPUT_LIMIT = 30_000

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
_SNIFF_BYTES = 8 * 1024


def truncate(s: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n[{len(s) - limit} characters truncated]"


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def is_binary(path: Path) -> bool:
    """True when the first 8 KB hold a NUL byte or are not UTF-8."""
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    if b"\x00" in head:
        return True
    try:
        # incremental decoding tolerates a multibyte character cut at the boundary
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        return True
    return False


def unified_diff(old: str, new: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
