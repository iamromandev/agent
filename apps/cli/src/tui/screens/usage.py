"""The `/usage` viewer: the session's token totals. Read-only and fail-safe: Escape or `q` closes it.

Names come from the model and the session file, so everything is a `rich.text.Text` built from the summary's
already-escaped strings, never markup."""

from __future__ import annotations

from typing import ClassVar

from core.engine.usage import EMPTY_TEXT, UsageSummary
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from src.tui.styles import ACCENT, DIM, MODIFIED

LABEL_WIDTH = 20


def render(summary: UsageSummary) -> Text:
    text = Text(no_wrap=False, overflow="fold")
    for label, value in summary.lines:
        text.append(f"{label:<{LABEL_WIDTH}}", ACCENT)
        text.append(f"{value}\n")
    if summary.empty:
        text.append(f"\n{EMPTY_TEXT}\n", DIM)
    if summary.models:
        text.append("\nBy model\n", ACCENT)
        for name, tokens_in, tokens_out, calls in summary.models:
            text.append(f"  {name}  in {tokens_in} · out {tokens_out} · {calls} calls\n")
    if summary.note:
        text.append(f"\n{summary.note}\n", MODIFIED)
    return text


class UsageScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    UsageScreen { align: center middle; }
    UsageScreen > Vertical {
        width: 90%;
        max-width: 100;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 0 1;
    }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
    ]

    def __init__(self, summary: UsageSummary) -> None:
        super().__init__()
        self.summary = summary

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(Text("Session usage"))
            yield Static(render(self.summary), id="usage")
            yield Label(Text("Esc or q close", style=DIM))
