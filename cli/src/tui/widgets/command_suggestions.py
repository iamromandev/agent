"""The popup above the input that suggests command names while a `/name` is typed."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from src.agent.text import escape


class CommandSuggestions(OptionList, can_focus=False):
    """Hidden until `update_for` finds a match. It never takes focus: the input keeps it and drives the popup."""

    DEFAULT_CSS = "CommandSuggestions { display: none; height: auto; max-height: 10; }"

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._names: list[str] = []

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    def update_for(self, text: str, entries: Sequence[tuple[str, str, str | None]]) -> None:
        """Show the entries `(name, description, source or None for a built-in)` whose name starts with the typed
        prefix; hide the popup when `text` is not a bare `/prefix` or nothing matches."""
        matches = (
            sorted(e for e in entries if e[0].startswith(text[1:]))
            if text.startswith("/") and not any(c.isspace() for c in text)
            else []
        )
        if not matches:
            self._hide()
            return
        self._names = [name for name, _, _ in matches]
        self.clear_options()
        self.add_options([Option(_row(*entry)) for entry in matches])
        self.highlighted = 0
        self.display = True

    def _hide(self) -> None:
        self._names = []
        self.display = False

    def hide(self) -> None:
        self._hide()

    def move(self, delta: int) -> None:
        if self._names:
            self.highlighted = max(0, min(len(self._names) - 1, (self.highlighted or 0) + delta))

    def name_at(self, index: int) -> str | None:
        return self._names[index] if 0 <= index < len(self._names) else None

    def chosen(self) -> str | None:
        index = self.highlighted
        return self._names[index] if index is not None and index < len(self._names) else None


MAX_DESCRIPTION = 60  # characters of a description shown: a row is one line


def _row(name: str, description: str, source: str | None) -> Text:
    line = f"/{escape(name)}"
    if description:
        shown = escape(" ".join(description.split()))
        line += f"  {shown if len(shown) <= MAX_DESCRIPTION else shown[: MAX_DESCRIPTION - 1] + '…'}"
    if source:
        line += f" ({source})"
    return Text(line, no_wrap=True, overflow="ellipsis")


