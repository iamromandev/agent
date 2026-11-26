"""The pasted images waiting for the next message, shown above the input as chips."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import Static

from src.engine.text import escape_display


class AttachmentPanel(Static):
    """Hidden when nothing is pending. Text is escaped and never read as markup."""

    DEFAULT_CSS = """
    AttachmentPanel {
        display: none; height: auto; max-height: 3;
        border-top: solid $panel;
        border-title-align: left;
        border-title-color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self.plain = ""

    def show(self, labels: Sequence[str]) -> None:
        """One chip per label: `[image N: <label>]`."""
        if not labels:
            self.plain = ""
            self.border_title = ""
            self.display = False
            return
        chips = " ".join(f"[image {n}: {escape_display(label)}]" for n, label in enumerate(labels, 1))
        text = Text(chips, style="dim", no_wrap=True, overflow="ellipsis")
        self.plain = text.plain
        self.update(text)
        self.border_title = f" images {len(labels)} "
        self.display = True
