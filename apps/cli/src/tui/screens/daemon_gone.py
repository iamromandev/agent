"""Asked when the `--shared` daemon did not come back: quit, or keep waiting. The input box is left as it is."""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


class DaemonGoneScreen(ModalScreen[bool]):
    """True: keep waiting; False: quit."""

    DEFAULT_CSS = """
    DaemonGoneScreen { align: center middle; }
    DaemonGoneScreen > Vertical {
        width: 70%;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 0 1;
    }
    DaemonGoneScreen OptionList { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "wait", "Keep waiting", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(Text(self.reason, "bold"))
            yield Label(Text("Your typed text stays in the input. Start it again with agent serve, or quit."))
            choices = OptionList(Option("Keep waiting", id="wait"), Option("Quit", id="quit"))
            choices.highlighted = 0  # a stray Enter does not quit
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "wait")

    def action_wait(self) -> None:
        self.dismiss(True)
