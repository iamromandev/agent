"""Small text helpers shared by the tools and the loop."""

from __future__ import annotations

import codecs
import difflib
import re
from pathlib import Path

TOOL_OUTPUT_LIMIT = 30_000

_OSC_BODY = r"[^\x07\x1b]{0,2048}"
_OSC_END = r"(?:\x07|\x1b\\)"
# Parameters are bounded, so a kept style token is at most 67 bytes. Longer sequences are not styles:
# the oversized alternatives of ESCAPE drop them whole (up to their terminator), so no body leaks as text.
# (One cut by a chunk boundary after more than 64 parameter bytes still leaks its tail.)
SGR = re.compile(r"\x1b\[[0-9;:]{0,64}m")
ESCAPE = re.compile(
    r"\x1b\[[0-9;:?]{0,64}[ -/]{0,16}[@-~]|\x1b\]" + _OSC_BODY + _OSC_END
    + r"|\x1b\[[0-9;:?]{65,}[ -/]{0,16}[@-~]?|\x1b\][^\x07\x1b]{2049,}(?:\x07|\x1b\\)?"
)
# A partial sequence at the very end of the text (anchored, with bounded runs, so it stays linear).
DANGLING = re.compile(r"\x1b(?:\[[0-9;:?]{0,64}[ -/]{0,16}|\]" + _OSC_BODY + r"\x1b?)?\Z")
_STRAY = re.compile(r"\x1b(?!\[[0-9;:]{0,64}m)")
_SNIFF_BYTES = 8 * 1024


def truncate(s: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n[{len(s) - limit} characters truncated]"


def strip_ansi(s: str) -> str:
    return ESCAPE.sub("", s)


def keep_sgr(s: str) -> str:
    """Remove every escape sequence except SGR (color and style); also a partial one at the end."""
    kept = ESCAPE.sub(lambda m: m.group() if SGR.fullmatch(m.group()) else "", s)
    return _STRAY.sub("", DANGLING.sub("", kept))


def truncate_ansi(s: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """`truncate` for text with SGR sequences: they do not count, and are never cut in two.

    `strip_ansi(truncate_ansi(s)) == truncate(strip_ansi(s))`.
    """
    visible = len(SGR.sub("", s))
    if visible <= limit:
        return s
    out: list[str] = []
    left = limit
    seen_sgr = False
    position = 0
    for match in SGR.finditer(s):
        text = s[position : match.start()]
        if len(text) >= left:
            out.append(text[:left])
            left = 0
            break
        out.append(text)
        left -= len(text)
        if left == 0:
            break
        out.append(match.group())
        seen_sgr = True
        position = match.end()
    else:
        out.append(s[position:][:left])
    if seen_sgr:
        out.append("\x1b[0m")
    out.append(f"\n[{visible - limit} characters truncated]")
    return "".join(out)


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
