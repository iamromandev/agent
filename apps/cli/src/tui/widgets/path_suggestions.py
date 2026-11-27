"""The popup above the input that suggests workspace paths while an `@token` is typed."""

from __future__ import annotations

from collections.abc import Sequence

from core.engine.mentions_index import Suggestion
from core.engine.text import escape
from rich.text import Text
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option


class PathSuggestions(OptionList, can_focus=False):
    """Hidden until `update_for` gets suggestions. It never takes focus: the input keeps it and drives the popup."""

    DEFAULT_CSS = "PathSuggestions { display: none; height: auto; max-height: 10; }"

    class Toggled(Message):
        """The list opened or closed; bubbles to the app, which refreshes the key hints."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._items: list[Suggestion] = []

    def _set_open(self, value: bool) -> None:
        if bool(self.display) != value:
            self.display = value
            self.post_message(self.Toggled())

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    def update_for(self, suggestions: Sequence[Suggestion]) -> None:
        if not suggestions:
            self.hide()
            return
        if self.is_open and list(suggestions) == self._items:
            return  # nothing changed: keep the rows and the highlight
        self._items = list(suggestions)
        self.clear_options()
        self.add_options(
            [Option(Text(escape(s.path), style="dim" if s.folder else "", no_wrap=True, overflow="ellipsis")) for s in self._items]
        )
        self.highlighted = 0
        self._set_open(True)

    def hide(self) -> None:
        self._items = []
        self._set_open(False)

    def move(self, delta: int) -> None:
        if self._items:
            self.highlighted = max(0, min(len(self._items) - 1, (self.highlighted or 0) + delta))

    def at(self, index: int) -> Suggestion | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def chosen(self) -> Suggestion | None:
        index = self.highlighted
        return self.at(index) if index is not None else None
