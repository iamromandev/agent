"""A bounded plain-text copy of a terminal's output: what `term_read` returns."""

from __future__ import annotations

import codecs
import re
from collections import deque

from src.agent.tools.text import DANGLING, ESCAPE

# C0 controls other than tab, newline and carriage return; also the stray ESC the sequences above leave.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clip(line: str, limit: int) -> str:
    """`line` cut to at most `limit` UTF-8 bytes, on a character."""
    if len(line) * 4 <= limit:
        return line
    if len(line) > limit:
        line = line[:limit]
    if len(line.encode()) <= limit:
        return line
    return line.encode()[:limit].decode("utf-8", errors="ignore")


class LineBuffer:
    """The last `max_lines` lines of a byte stream, as text without escape sequences.

    Bytes arrive in arbitrary chunks, so a UTF-8 character, an escape sequence or a CRLF may be cut in two:
    each is completed by the next `feed`. A lone carriage return (a progress bar redrawing its line) keeps
    only the text after it. Memory is bounded by `max_lines * max_line_bytes`.
    """

    def __init__(self, max_lines: int = 5_000, max_line_bytes: int = 4_096) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._max_line_bytes = max_line_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._held = ""  # an escape sequence cut by the end of the last feed
        self._partial = ""  # the line being written
        self._cr = False  # the last feed ended with a carriage return

    def __len__(self) -> int:
        return len(self._lines)

    def feed(self, data: bytes) -> None:
        text = self._held + self._decoder.decode(data)
        dangling = DANGLING.search(text)
        self._held = text[dangling.start() :] if dangling else ""
        if dangling:
            text = text[: dangling.start()]
        text = _CONTROL.sub("", ESCAPE.sub("", text))
        if text:  # a feed with nothing to show (empty, or only part of an escape) changes nothing
            if self._cr and not text.startswith("\n"):
                self._partial = ""  # the carriage return was alone: the line is rewritten
            self._cr = text.endswith("\r")
            if self._cr:
                text = text[:-1]
        *complete, last = text.replace("\r\n", "\n").split("\n")
        for piece in complete:
            self._lines.append(self._join(piece))
            self._partial = ""
        self._partial = self._join(last)

    def _join(self, piece: str) -> str:
        line = (self._partial + piece).rpartition("\r")[2]
        return _clip(line, self._max_line_bytes)

    def tail(self, n: int) -> list[str]:
        """At most the last `n` lines, the line being written (when not empty) counting as the last."""
        if n <= 0:
            return []
        lines = list(self._lines)
        if self._partial:
            lines.append(self._partial)
        return lines[-n:]
