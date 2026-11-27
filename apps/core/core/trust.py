"""The launch trust prompts without a UI: what each one shows (`TrustAsk`) and what a yes or a no does
(`apply_trust`). The TUI's three screens and, later, the session server's `Ask` messages are built on this."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from core.engine.configguard import MAX_SHOWN, NEXT_LAUNCH, HeldGlobal, MarkerInfo
from core.engine.configwatch import TrustRevoked
from core.engine.hooks.config import HookEntry
from core.engine.mcp.config import ServerConfig
from core.engine.permissions.rules import Rule
from core.engine.text import escape

TrustKind = Literal["global_trust", "mcp_trust", "permission_trust"]
Level = Literal["info", "warn"]

_SPACES = re.compile(r" {2,}")


@dataclass(frozen=True)
class TrustAsk:
    kind: TrustKind
    lines: list[str]  # one per item, escaped and ready to show
    note: str | None = None


@dataclass
class TrustOutcome:
    notices: list[tuple[Level, str]] = field(default_factory=list)
    start: list[str] = field(default_factory=list)  # MCP servers to start now (on the caller's event loop)


class GlobalGuard(Protocol):
    def trust_global(self) -> None: ...


class ProjectPermissions(Protocol):
    def trust_project(self) -> None: ...


def visible(text: str) -> str:
    """`text` as a file wrote it, shown whole: odd characters escaped, and each run of 2+ spaces written as a space
    sign and a count, so padding cannot push the rest of a command out of sight."""
    return _SPACES.sub(lambda run: f"\u2420\u00d7{len(run.group())}", escape(text))


def command_line(server: ServerConfig) -> str:
    """The full command line as the shell would see it (quoted), so odd arguments are visible."""
    return shlex.join([escape(server.command), *(escape(arg) for arg in server.args)])


def env_note(server: ServerConfig) -> str:
    return f"(env: {', '.join(escape(key) for key in server.env)})" if server.env else ""


def sort_rules(rules: Sequence[Rule]) -> list[Rule]:
    """What allows the most comes first: a bare tool, a prefix; then the exact calls, in the file's order."""
    return sorted(rules, key=lambda rule: not (rule.value is None or rule.prefix))


def hook_line(hook: HookEntry) -> str:
    return f"hook {escape(hook.label)} (timeout {hook.timeout:g}s) {visible(hook.command)}"


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


def mcp_line(server: ServerConfig) -> str:
    name = escape(server.name)
    env = f"  {env_note(server)}" if server.env else ""
    over = f"\n  overrides the global '{name}'" if server.overrides_global else ""
    return f"{name}  {command_line(server)}{env}{over}"


def trust_lines(
    kind: TrustKind,
    *,
    rules: Sequence[Rule] = (),
    hooks: Sequence[HookEntry] = (),
    servers: Sequence[ServerConfig] = (),
    marker: MarkerInfo | None = None,
    deny: Sequence[Rule] = (),
    held: HeldGlobal | None = None,
) -> TrustAsk:
    """What the prompt of `kind` lists, in the order the screens show it."""
    if kind == "mcp_trust":
        return TrustAsk(kind, [mcp_line(server) for server in servers])
    lines = [f"allow {visible(rule.text)}" for rule in sort_rules(rules)]
    lines += [hook_line(hook) for hook in hooks]
    if kind == "permission_trust":
        return TrustAsk(kind, lines)
    held = held if held is not None else HeldGlobal()
    for server in servers:
        env = f" {env_note(server)}" if server.env else ""
        lines.append(f"mcp {visible(server.name)}  {visible(command_line(server))}{env}")
    lines += [f"keepEnv {visible(name)}" for name in held.keep_env]
    for label, names in held.kinds():
        lines += [f"{label} {visible(name)}" for name in names]
    lines += [f"deny rule {visible(rule.text)} (applies either way)" for rule in deny]
    return TrustAsk(kind, lines, marker_text(marker))


def pending_trust(
    *,
    global_waiting: bool,
    marker: MarkerInfo | None,
    global_rules: Sequence[Rule],
    global_hooks: Sequence[HookEntry],
    global_servers: Sequence[ServerConfig],
    global_deny: Sequence[Rule],
    held: HeldGlobal,
    project_servers: Sequence[ServerConfig],
    project_rules: Sequence[Rule],
    project_hooks: Sequence[HookEntry],
) -> list[TrustAsk]:
    """The prompts to show at launch, in the order they are asked: global, MCP, permissions."""
    asks: list[TrustAsk] = []
    if global_waiting:
        asks.append(
            trust_lines(
                "global_trust",
                rules=global_rules,
                hooks=global_hooks,
                servers=global_servers,
                marker=marker,
                deny=global_deny,
                held=held,
            )
        )
    if project_servers:
        asks.append(trust_lines("mcp_trust", servers=project_servers))
    if project_rules or project_hooks:
        asks.append(trust_lines("permission_trust", rules=project_rules, hooks=project_hooks))
    return asks


def apply_trust(
    kind: TrustKind,
    yes: bool,
    *,
    guard: GlobalGuard | None = None,
    permissions: ProjectPermissions | None = None,
    mcp_trust: Callable[[], None] | None = None,
    servers: Sequence[str] = (),
    held_any: bool = False,
    has_rules: bool = True,
    has_hooks: bool = False,
) -> TrustOutcome:
    """What a yes or a no does. Blocking (writes the trust stores): call it off the event loop. The caller shows the
    notices and starts `start` (MCP servers) itself."""
    out = TrustOutcome()
    if kind == "global_trust":
        if not yes:
            out.notices.append(
                ("info", "global config not trusted: global allow rules, hooks and MCP servers are off for this run")
            )
            return out
        if guard is None:
            return out
        try:
            guard.trust_global()  # removes the marker
        except OSError as exc:  # TrustRevoked too: nothing was turned on, the marker stays
            out.notices.append(("warn", escape(f"global config: {exc}")))
            return out
        out.start = list(servers)
        later = f"; {NEXT_LAUNCH}" if held_any else ""
        out.notices.append(
            ("info", f"global config trusted: global allow rules, hooks and MCP servers are on{later}")
        )
    elif kind == "mcp_trust":
        if not yes:
            out.notices.append(("info", f"mcp: project servers not trusted, so not started: {', '.join(servers)}"))
            return out
        if mcp_trust is not None:
            try:
                mcp_trust()
            except TrustRevoked as exc:  # project trust was revoked meanwhile: nothing starts
                out.notices.append(("warn", escape(f"mcp: {exc}")))
                return out
            except OSError as exc:  # the servers still start for this run; it will ask again next time
                out.notices.append(("warn", f"mcp: could not remember the trust: {exc}"))
        out.start = list(servers)
    else:
        if not yes:
            if has_rules and has_hooks:
                text = "project permission rules and hooks not trusted: allow rules and hooks are off for this run"
            elif has_hooks:
                text = "project hooks not trusted: hooks are off for this run"
            else:
                text = "project permission rules not trusted: allow rules are off for this run"
            out.notices.append(("info", text))
            return out
        if permissions is None:
            return out
        try:
            permissions.trust_project()
        except TrustRevoked as exc:  # the config watch revoked project trust meanwhile: nothing was trusted
            out.notices.append(("warn", escape(f"permissions: {exc}")))
        except OSError as exc:  # the rules are on for this run; it will ask again next time
            out.notices.append(("warn", escape(f"permissions: could not remember the trust: {exc}")))
    return out
