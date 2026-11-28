"""The launch prompt for the global config after an earlier run saw a command change it (the config watch's marker).
Fail-safe like the project prompts: Escape, or a stray Enter, keeps the global allow rules, hooks and MCP servers off."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.configguard import HeldGlobal, MarkerInfo
from core.engine.hooks.config import HookEntry
from core.engine.lsp.config import ServerSpec
from core.engine.mcp.config import ServerConfig
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
        lsp: Sequence[ServerSpec] = (),
    ) -> None:
        super().__init__()
        # What allows the most comes first, as in the project prompt.
        self.rules = sort_rules(rules)
        self.hooks = list(hooks)
        self.servers = list(servers)
        self.lsp = list(lsp)
        self.marker = marker
        self.deny = list(deny)  # shown for the full picture: deny rules apply either way
        self.held = held if held is not None else HeldGlobal()

    def compose(self) -> ComposeResult:
        ask = trust_lines(
            "global_trust",
            rules=self.rules,
            hooks=self.hooks,
            servers=self.servers,
            marker=self.marker,
            deny=self.deny,
            held=self.held,
            lsp=self.lsp,
        )
        with Vertical():
            yield Label(Text(ask.title, "bold"), id="title")
            for line in ask.header:
                yield Label(Text(line))  # a `Text`, never markup
            with VerticalScroll(id="items"):
                for line in ask.lines:
                    yield Label(Text(line))
            for line in ask.footer:
                yield Label(Text(line))
            # "Don't trust" comes first and is highlighted from the start: a stray Enter must never allow anything.
            choices = OptionList(Option(ask.decline, id="decline"), Option(ask.accept, id="trust"))
            choices.highlighted = 0
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "trust")

    def action_decline(self) -> None:
        self.dismiss(False)
