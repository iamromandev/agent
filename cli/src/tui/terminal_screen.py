"""The emulator screen of a terminal tab: pyte's screen plus the alternate screen, which pyte 0.8 lacks."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pyte
from pyte import charsets as cs
from pyte.screens import Cursor, StaticDefaultDict

_ALTERNATE = (47, 1047, 1049)  # xterm's ways to switch to the alternate screen (private modes)
_DECCKM = 1 << 5  # application cursor keys; pyte keeps a private mode `n` in `mode` as `n << 5`
_BRACKETED_PASTE = 2004 << 5
_DECCOLM = 3  # private mode: 80/132 columns. xterm ignores it unless allowed, and the pane's size is its own


class TerminalStream(pyte.ByteStream):
    """A `pyte.ByteStream` that skips a sequence its screen cannot handle instead of losing the rest of the chunk.

    pyte lets the error of a handler (`ESC [ ; 5 C` is a TypeError, `ESC [ 6 n` in origin mode an assertion)
    escape `feed`, so everything after it in the same chunk would never be drawn. Programs are free to
    send what a real terminal would ignore.
    """

    def _send_to_parser(self, data: str) -> bool | None:
        try:
            return super()._send_to_parser(data)  # the parser is reset when it raises
        except Exception:
            return False  # the sequence is dropped; the next character starts a new one


class TerminalScreen(pyte.Screen):
    """A `pyte.Screen` that also has the alternate screen (`ESC [ ? 1049 h`) full-screen programs draw on.

    While it is on, `buffer` and `cursor` are the alternate screen's; the main ones wait in `_saved` and
    come back, resized with the screen, when the program leaves it.
    """

    _saved: tuple[Any, Cursor] | None = None
    reply: Callable[[str], None] | None = None  # where the answers to the program's questions go (cursor position, ...)

    @property
    def application_cursor(self) -> bool:
        """True while the program asked for arrows as `ESC O A` instead of `ESC [ A`."""
        return _DECCKM in self.mode

    @property
    def bracketed_paste(self) -> bool:
        """True while the program asked for pasted text between `ESC [ 200 ~` and `ESC [ 201 ~`."""
        return _BRACKETED_PASTE in self.mode

    def draw(self, data: str) -> None:
        # pyte stops drawing a run of text at its first zero-width character that does not combine (an emoji
        # selector, a joiner), losing the rest of the run. Such characters add nothing to a cell: leave them out.
        super().draw("".join(c for c in data if pyte.screens.wcwidth(c) > 0 or unicodedata.combining(c)))

    def restart_rendition(self) -> None:
        """Back to plain text: default colors and attributes, the default charsets (as after `ESC [ 0 m`, `SI`)."""
        self.select_graphic_rendition(0)
        self.charset = 0
        self.g0_charset = cs.LAT1_MAP
        self.g1_charset = cs.VT100_MAP

    def write_process_input(self, data: str) -> None:
        if self.reply is not None:
            self.reply(data)

    def reset(self) -> None:
        super().reset()
        self._saved = None  # a full reset leaves the alternate screen without bringing the main one back

    def set_mode(self, *modes: int, **kwargs: Any) -> None:
        private = kwargs.get("private")
        if private:
            modes = tuple(mode for mode in modes if mode != _DECCOLM)  # pyte would resize and clear the screen
        super().set_mode(*modes, **kwargs)
        if private and any(mode in _ALTERNATE for mode in modes):
            self._enter_alternate()

    def reset_mode(self, *modes: int, **kwargs: Any) -> None:
        private = kwargs.get("private")
        if private:
            modes = tuple(mode for mode in modes if mode != _DECCOLM)
        super().reset_mode(*modes, **kwargs)
        if private and any(mode in _ALTERNATE for mode in modes):
            self._leave_alternate()

    def resize(self, lines: int | None = None, columns: int | None = None) -> None:
        before = (self.lines, self.columns)
        self._resize(lines, columns)
        after = (self.lines, self.columns)
        if self._saved is not None and after != before:  # the main screen is resized too
            self._swap()
            self.lines, self.columns = before
            self._resize(*after)
            self._swap()

    def _resize(self, lines: int | None, columns: int | None) -> None:
        """Resize the current screen the way xterm does: fewer rows drop the top ones, as far as the cursor needs.

        pyte drops rows from the top whatever the cursor is, which can leave the cursor below the last row.
        Here only as many rows go from the top as it takes to keep the cursor's row, and the rest is cut
        from the bottom.
        """
        lines = lines or self.lines
        if lines < self.lines:
            drop = max(0, self.cursor.y - (lines - 1))
            rows = dict(self.buffer)
            self.buffer.clear()
            for y, line in rows.items():
                if 0 <= y - drop < lines:
                    self.buffer[y - drop] = line
            self.cursor.y -= drop
            self.lines = lines  # the rows are gone: pyte has only the columns left to do
            self.dirty.update(range(lines))
            self.set_margins()
        super().resize(lines, columns)
        self.cursor.y = min(self.cursor.y, self.lines - 1)
        self.cursor.x = min(self.cursor.x, self.columns - 1)

    def _swap(self) -> None:
        assert self._saved is not None
        (buffer, cursor), self._saved = self._saved, (self.buffer, self.cursor)
        cursor.hidden = self.cursor.hidden  # the cursor's visibility is the terminal's, not the screen's
        self.buffer, self.cursor = buffer, cursor

    def _enter_alternate(self) -> None:
        if self._saved is not None:
            return
        cursor = Cursor(self.cursor.x, self.cursor.y, self.cursor.attrs)
        cursor.hidden = self.cursor.hidden  # visibility carries over; `_swap` carries it back
        self._saved = (self.buffer, self.cursor)
        self.buffer = defaultdict(lambda: StaticDefaultDict(self.default_char))
        self.cursor = cursor
        self.dirty.update(range(self.lines))

    def _leave_alternate(self) -> None:
        if self._saved is None:
            return
        self._swap()
        self._saved = None
        self.dirty.update(range(self.lines))
