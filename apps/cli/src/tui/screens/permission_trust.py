"""The prompt for a project's permission rules and hooks. Fail-safe: Escape, or a stray Enter, does not trust them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.hooks.config import HookEntry
from core.engine.permissions.rules import Rule
from core.trust import sort_rules, trust_lines
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


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
    PermissionTrustScreen Label { width: 100%; }
    PermissionTrustScreen #rules { height: auto; max-height: 15; overflow-x: hidden; }
    PermissionTrustScreen #rules:focus { border-left: thick $accent; }
    PermissionTrustScreen OptionList { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "decline", "Don't trust", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, rules: Sequence[Rule], hooks: Sequence[HookEntry] = ()) -> None:
        super().__init__()
        self.hooks = list(hooks)
        # What allows the most comes first: a bare tool, a prefix; then the exact calls, in the file's order.
        self.rules = sort_rules(rules)

    def compose(self) -> ComposeResult:
        with Vertical():
            parts = []
            if self.rules:
                parts.append(f"{len(self.rules)} allow rule{'s' if len(self.rules) != 1 else ''}")
            if self.hooks:
                parts.append(f"{len(self.hooks)} hook{'s' if len(self.hooks) != 1 else ''}")
            yield Label(Text(f"Trust this project's settings? ({', '.join(parts)})", "bold"), id="title")
            if self.rules:
                yield Label(Text(".agent/settings.json in this folder wants to allow these calls without asking:"))
            if self.hooks:
                yield Label(Text(".agent/settings.json in this folder wants to run these commands at these events:"))
            with VerticalScroll(id="rules"):
                for line in trust_lines("permission_trust", rules=self.rules, hooks=self.hooks).lines:
                    yield Label(Text(line))  # a `Text`, never markup; odd characters escaped, wrapped
            if self.rules:
                yield Label(Text("Matching calls will run without asking you. Tab scrolls the list."))
            if self.hooks:
                yield Label(Text("Hooks run in this folder like bash, and can refuse your tool calls. Tab scrolls the list."))
            # "Don't trust" comes first and is highlighted from the start: a stray Enter must never allow a call.
            what = " and ".join(name for name, items in (("rules", self.rules), ("hooks", self.hooks)) if items)
            choices = OptionList(Option("Don't trust (keep them off)", id="decline"), Option(f"Trust these {what}", id="trust"))
            choices.highlighted = 0
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "trust")

    def action_decline(self) -> None:
        self.dismiss(False)
