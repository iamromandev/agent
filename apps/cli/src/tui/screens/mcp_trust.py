"""The prompt for a project's MCP servers. Fail-safe: Escape, or a stray Enter, does not trust them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.mcp.config import ServerConfig
from core.engine.text import escape
from core.trust import command_line, env_note
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


def describe(server: ServerConfig) -> Text:
    """One server: its name, the command line, the names of the env vars it sets, and whether it replaces a global
    server. A `Text`, never markup, with every odd character escaped."""
    text = Text()
    name = escape(server.name)
    text.append(name, "bold")
    text.append(f"  {command_line(server)}")
    if server.env:
        text.append(f"  {env_note(server)}", "dim")
    if server.overrides_global:
        text.append(f"\n  overrides the global '{name}'", "dim")
    return text


class McpTrustScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    McpTrustScreen { align: center middle; }
    McpTrustScreen > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 0 1;
    }
    McpTrustScreen #servers { height: auto; max-height: 15; }
    McpTrustScreen #servers:focus { border-left: thick $accent; }
    McpTrustScreen OptionList { height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "decline", "Don't trust", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, servers: Sequence[ServerConfig]) -> None:
        super().__init__()
        self.servers = list(servers)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(Text(".agent/mcp.json in this folder wants to start these MCP servers on your machine:"))
            with VerticalScroll(id="servers"):
                for server in self.servers:
                    yield Label(describe(server))
            yield Label(Text("Files these commands run (scripts, packages) are not checked. Tab scrolls the list."))
            # "Don't trust" comes first and is highlighted from the start: a stray Enter must never start a process.
            choices = OptionList(Option("Don't trust (keep them stopped)", id="decline"), Option("Trust and start them", id="trust"))
            choices.highlighted = 0
            yield choices

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "trust")

    def action_decline(self) -> None:
        self.dismiss(False)
