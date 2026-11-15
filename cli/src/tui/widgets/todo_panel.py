"""The todo list pinned above the input."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import Static

from src.engine.text import escape
from src.engine.todo import MARK, TodoItem, active


class TodoPanel(Static):
    """Hidden unless the list has something left to do. Item text is escaped and never read as markup."""

    DEFAULT_CSS = "TodoPanel { display: none; height: auto; max-height: 12; border-top: solid $panel; padding: 0 1; }"

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self.plain = ""  # what is drawn, as text (for tests and for the empty case)

    def show(self, items: Sequence[TodoItem]) -> None:
        if not active(items):
            self.plain = ""
            self.display = False
            return
        text = Text(no_wrap=True, overflow="ellipsis")
        for number, item in enumerate(items):
            if number:
                text.append("\n")
            style = "bold" if item.status == "in_progress" else "dim" if item.status == "done" else ""
            text.append(f"{MARK[item.status]} {escape(item.content)}", style=style)
        self.plain = text.plain
        self.update(text)
        self.display = True
