"""The launch prompt for the global config after an earlier run saw a command change it (the config watch's marker).
Fail-safe like the project prompts: Escape, or a stray Enter, keeps the global allow rules, hooks and MCP servers off."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.configguard import NEXT_LAUNCH, HeldGlobal, MarkerInfo
from core.engine.hooks.config import HookEntry
from core.engine.mcp.config import ServerConfig
from core.engine.permissions.rules import Rule
from core.trust import marker_text, sort_rules, trust_lines
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'s' if n != 1 else ''}"


class GlobalTrustScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    GlobalTrustScreen { align: center middle; }
    GlobalTrustScreen > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 0 1;
    }
    GlobalTrustScreen Label { width: 100%; }
    GlobalTrustScreen #items { height: auto; max-height: 15; overflow-x: hidden; }
    GlobalTrustScreen #items:focus { border-left: thick $accent; }
    GlobalTrustScreen OptionList { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "decline", "Don't trust", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(
        self,
        rules: Sequence[Rule],
        hooks: Sequence[HookEntry],
        servers: Sequence[ServerConfig],
        marker: MarkerInfo | None,
        *,
        deny: Sequence[Rule] = (),
        held: HeldGlobal | None = None,
    ) -> None:
        super().__init__()
        # What allows the most comes first, as in the project prompt.
        self.rules = sort_rules(rules)
        self.hooks = list(hooks)
        self.servers = list(servers)
        self.marker = marker
        self.deny = list(deny)  # shown for the full picture: deny rules apply either way
        self.held = held if held is not None else HeldGlobal()

    def compose(self) -> ComposeResult:
        with Vertical():
            parts = [
                _count(len(items), what)
                for items, what in ((self.rules, "allow rule"), (self.hooks, "hook"), (self.servers, "MCP server"))
                if items
            ]
            held = self.held
            parts += [_count(len(names), f"global {kind}") for kind, names in held.kinds()]
            if held.keep_env:
                parts.append(_count(len(held.keep_env), "keepEnv name"))
            title = "Trust the global config again?" + (f" ({', '.join(parts)})" if parts else "")
            yield Label(Text(title, "bold"), id="title")
            yield Label(Text(marker_text(self.marker)))  # a `Text`, never markup
            if parts:
                yield Label(Text("Until you trust it, these stay off (the files may hold what the command added):"))
            else:
                yield Label(Text("It holds no allow rules, hooks or MCP servers now."))
            with VerticalScroll(id="items"):
                lines = trust_lines(
                    "global_trust",
                    rules=self.rules,
                    hooks=self.hooks,
                    servers=self.servers,
                    deny=self.deny,
                    held=held,
                )
                for line in lines.lines:
                    yield Label(Text(line))
            yield Label(Text("Deny rules and protected paths apply either way. Tab scrolls the list."))
            if held.any:
                yield Label(Text(f"Trust turns the allow rules, hooks and MCP servers on now; {NEXT_LAUNCH}."))
            # "Don't trust" comes first and is highlighted from the start: a stray Enter must never allow anything.
            choices = OptionList(
                Option("Don't trust (keep them off; the next launch asks again)", id="decline"),
                Option("Trust the global config", id="trust"),
            )
            choices.highlighted = 0
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "trust")

    def action_decline(self) -> None:
        self.dismiss(False)
