"""The dim line under the input that lists the keys for the current state."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class KeyHints(Static, can_focus=False):
    """Display only: it never takes focus, and the app decides the text (`src.tui.hints.hint_text`)."""

    DEFAULT_CSS = """KeyHints { height: 1; padding: 0 2; color: $text-muted; background: transparent;
        text-wrap: nowrap; text-overflow: ellipsis; }"""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self.plain = ""

    def show(self, text: str) -> None:
        self.plain = text
        self.update(Text(text, no_wrap=True, overflow="ellipsis"))
