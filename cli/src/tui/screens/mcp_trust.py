"""The prompt for a project's MCP servers. Fail-safe: Escape, or a stray Enter, does not trust them."""

from __future__ import annotations

import shlex
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

from src.agent.mcp.config import ServerConfig


def escape(text: str) -> str:
    """`text` with everything but printable ASCII written out (`\\x1b`, `\\u202e`, `\\U0001f600`).

    A project file is written by someone else: it must not be able to hide or rewrite what is shown here
    with escape sequences, carriage returns, or characters that reorder the text.
    """
    out: list[str] = []
    for char in text:
        code = ord(char)
        if char.isascii() and char.isprintable():
            out.append(char)
        elif code < 0x100:
            out.append(f"\\x{code:02x}")
        elif code < 0x10000:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def command_line(server: ServerConfig) -> str:
    """The full command line as the shell would see it (quoted), so odd arguments are visible."""
    return shlex.join([escape(server.command), *(escape(arg) for arg in server.args)])


def describe(server: ServerConfig) -> Text:
    """One server: its name, the command line, the names of the env vars it sets, and whether it replaces a global
    server. A `Text`, never markup, with every odd character escaped."""
    text = Text()
    name = escape(server.name)
    text.append(name, "bold")
    text.append(f"  {command_line(server)}")
    if server.env:
        text.append(f"  (env: {', '.join(escape(key) for key in server.env)})", "dim")
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
