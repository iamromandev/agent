"""The message box: Enter sends, ctrl+j or shift+enter adds a line."""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import TextArea


class ChatInput(TextArea):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+j", "newline", "New line", show=False),
        Binding("shift+enter", "newline", "New line", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def action_newline(self) -> None:
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":  # TextArea would insert a newline
            event.stop()
            event.prevent_default()
            text = self.text
            if text.strip():
                self.clear()
                self.post_message(self.Submitted(text))
            return
        await super()._on_key(event)
