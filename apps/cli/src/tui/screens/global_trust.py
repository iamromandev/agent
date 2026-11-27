"""The launch prompt for the global config after an earlier run saw a command change it (the config watch's marker).
Fail-safe like the project prompts: Escape, or a stray Enter, keeps the global allow rules, hooks and MCP servers off."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.configguard import MAX_SHOWN, NEXT_LAUNCH, HeldGlobal, MarkerInfo
from core.engine.hooks.config import HookEntry
from core.engine.mcp.config import ServerConfig
from core.engine.permissions.rules import Rule
from core.engine.text import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from src.tui.screens.mcp_trust import command_line
from src.tui.screens.permission_trust import visible


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'s' if n != 1 else ''}"


def marker_text(marker: MarkerInfo | None) -> str:
    """When and which files, as the marker recorded them (escaped; at most MAX_SHOWN names, then a count)."""
    if marker is None:
        return "A command changed the agent's global config during an earlier run."
    when = f" ({marker.at.astimezone().strftime('%Y-%m-%d %H:%M:%S')})" if marker.at is not None else ""
    text = f"A command changed the agent's global config during an earlier run{when}"
    if not marker.files:
        return text + "."
    shown = ", ".join(visible(name) for name in marker.files[:MAX_SHOWN])
    more = len(marker.files) - MAX_SHOWN
    return f"{text}: {shown}{f' and {more} more' if more > 0 else ''}."


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
        self.rules = sorted(rules, key=lambda rule: not (rule.value is None or rule.prefix))
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
                for rule in self.rules:
                    yield Label(Text(f"allow {visible(rule.text)}"))
                for hook in self.hooks:
                    yield Label(Text(f"hook {escape(hook.label)} (timeout {hook.timeout:g}s) {visible(hook.command)}"))
                for server in self.servers:
                    env = f" (env: {', '.join(escape(key) for key in server.env)})" if server.env else ""
                    yield Label(Text(f"mcp {visible(server.name)}  {visible(command_line(server))}{env}"))
                for name in held.keep_env:
                    yield Label(Text(f"keepEnv {visible(name)}"))
                for kind, names in held.kinds():
                    for name in names:
                        yield Label(Text(f"{kind} {visible(name)}"))
                for rule in self.deny:
                    yield Label(Text(f"deny rule {visible(rule.text)} (applies either way)"))
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
