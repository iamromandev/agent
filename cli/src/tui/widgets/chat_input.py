"""The message box: Enter sends, ctrl+j or shift+enter adds a line."""

from __future__ import annotations

from typing import Any, ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import TextArea

from src.tui.widgets.command_suggestions import CommandSuggestions

POPUP_KEYS = frozenset({"up", "down", "tab", "enter", "escape"})


class ChatInput(TextArea):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+j", "newline", "New line", show=False),
        Binding("shift+enter", "newline", "New line", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self.suggestions: CommandSuggestions | None = None  # set by the app; its keys are ours while it is open

    @property
    def popup_open(self) -> bool:
        return self.suggestions is not None and self.suggestions.is_open

    def complete(self, name: str) -> None:
        """Put `/name ` in the box, ready for arguments."""
        self.load_text(f"/{name} ")
        self.move_cursor(self.document.end)
        if self.suggestions is not None:
            self.suggestions.hide()

    def action_newline(self) -> None:
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        popup = self.suggestions
        if popup is not None and popup.is_open and event.key == "enter" and self.text == f"/{popup.chosen()}":
            popup.hide()  # the whole name is typed: Enter sends it, as it always did
        elif popup is not None and popup.is_open and event.key in POPUP_KEYS:
            event.stop()
            event.prevent_default()
            match event.key:
                case "up" | "down":
                    popup.move(-1 if event.key == "up" else 1)
                case "tab" | "enter":
                    if (name := popup.chosen()) is not None:
                        self.complete(name)
                    popup.hide()
                case _:  # escape
                    popup.hide()
            return
        if event.key == "enter":  # TextArea would insert a newline
            event.stop()
            event.prevent_default()
            text = self.text
            if text.strip():
                self.clear()
                self.post_message(self.Submitted(text))
            return
        await super()._on_key(event)
