"""The queued messages shown above the input."""

from __future__ import annotations

from collections.abc import Sequence

from core.host.queue import Queued
from rich.text import Text
from textual.widgets import Static

from src.tui.queue import panel_lines


class QueuePanel(Static):
    """Hidden when the queue is empty. Text is escaped by `panel_lines` and never read as markup."""

    DEFAULT_CSS = """
    QueuePanel {
        display: none; height: auto; max-height: 7;
        border-top: solid $panel;
        border-title-align: left;
        border-title-color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self.plain = ""  # what is drawn, as text (for tests and for the empty case)

    def show(self, entries: Sequence[Queued]) -> None:
        if not entries:
            self.plain = ""
            self.border_title = ""
            self.display = False
            return
        text = Text("\n".join(panel_lines(entries)), style="dim", no_wrap=True, overflow="ellipsis")
        self.plain = text.plain
        self.update(text)
        self.border_title = f" queued {len(entries)} "
        self.display = True
