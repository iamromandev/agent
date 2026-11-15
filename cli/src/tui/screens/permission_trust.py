"""The prompt for a project's permission rules. Fail-safe: Escape, or a stray Enter, does not trust them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from src.agent.permissions.rules import Rule
from src.agent.text import escape


class PermissionTrustScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    PermissionTrustScreen { align: center middle; }
    PermissionTrustScreen > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 0 1;
    }
    PermissionTrustScreen #rules { height: auto; max-height: 15; }
    PermissionTrustScreen #rules:focus { border-left: thick $accent; }
    PermissionTrustScreen OptionList { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "decline", "Don't trust", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, rules: Sequence[Rule]) -> None:
        super().__init__()
        # What allows the most comes first: a bare tool, a prefix; then the exact calls, in the file's order.
        self.rules = sorted(rules, key=lambda rule: not (rule.value is None or rule.prefix))

    def compose(self) -> ComposeResult:
        with Vertical():
            count = len(self.rules)
            title = Text(f"Trust this project's permission rules? ({count} allow rule{'s' if count != 1 else ''})", "bold")
            yield Label(title, id="title")
            yield Label(Text(".agent/settings.json in this folder wants to allow these calls without asking:"))
            with VerticalScroll(id="rules"):
                for rule in self.rules:
                    yield Label(Text(f"allow {escape(rule.text)}"))  # a `Text`, never markup; odd characters escaped
            yield Label(Text("Matching calls will run without asking you. Tab scrolls the list."))
            # "Don't trust" comes first and is highlighted from the start: a stray Enter must never allow a call.
            choices = OptionList(Option("Don't trust (keep them off)", id="decline"), Option("Trust these rules", id="trust"))
            choices.highlighted = 0
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "trust")

    def action_decline(self) -> None:
        self.dismiss(False)
