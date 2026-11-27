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
    """What a trust prompt shows, top to bottom: `title`, the `header` lines, the items (`lines`), the `footer` lines,
    then the two choices (`decline` first: a stray Enter must never trust). All of it escaped and ready to show."""

    kind: TrustKind
    lines: list[str]  # one per item, escaped and ready to show
    note: str | None = None  # the global prompt: when and which files the marker names (also the header's first line)
    title: str = ""
    header: list[str] = field(default_factory=list)
    footer: list[str] = field(default_factory=list)
    decline: str = ""
    accept: str = ""


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
    """What the prompt of `kind` lists, in the order the screens show it, with its title, texts and choices."""
    if kind == "mcp_trust":
        return TrustAsk(
            kind,
            [mcp_line(server) for server in servers],
            header=[".agent/mcp.json in this folder wants to start these MCP servers on your machine:"],
            footer=["Files these commands run (scripts, packages) are not checked. Tab scrolls the list."],
            decline="Don't trust (keep them stopped)",
            accept="Trust and start them",
        )
    lines = [f"allow {visible(rule.text)}" for rule in sort_rules(rules)]
    lines += [hook_line(hook) for hook in hooks]
    if kind == "permission_trust":
        parts = [_count(len(items), what) for items, what in ((rules, "allow rule"), (hooks, "hook")) if items]
        header, footer = [], []
        if rules:
            header.append(".agent/settings.json in this folder wants to allow these calls without asking:")
            footer.append("Matching calls will run without asking you. Tab scrolls the list.")
        if hooks:
            header.append(".agent/settings.json in this folder wants to run these commands at these events:")
            footer.append("Hooks run in this folder like bash, and can refuse your tool calls. Tab scrolls the list.")
        what = " and ".join(name for name, items in (("rules", rules), ("hooks", hooks)) if items)
        return TrustAsk(
            kind,
            lines,
            title=f"Trust this project's settings? ({', '.join(parts)})",
            header=header,
            footer=footer,
            decline="Don't trust (keep them off)",
            accept=f"Trust these {what}",
        )
    held = held if held is not None else HeldGlobal()
    parts = [_count(len(items), what) for items, what in ((rules, "allow rule"), (hooks, "hook"), (servers, "MCP server")) if items]
    parts += [_count(len(names), f"global {label}") for label, names in held.kinds()]
    if held.keep_env:
        parts.append(_count(len(held.keep_env), "keepEnv name"))
    note = marker_text(marker)
    header = [
        note,
        "Until you trust it, these stay off (the files may hold what the command added):"
        if parts
        else "It holds no allow rules, hooks or MCP servers now.",
    ]
    footer = ["Deny rules and protected paths apply either way. Tab scrolls the list."]
    if held.any:
        footer.append(f"Trust turns the allow rules, hooks and MCP servers on now; {NEXT_LAUNCH}.")
    for server in servers:
        env = f" {env_note(server)}" if server.env else ""
        lines.append(f"mcp {visible(server.name)}  {visible(command_line(server))}{env}")
    lines += [f"keepEnv {visible(name)}" for name in held.keep_env]
    for label, names in held.kinds():
        lines += [f"{label} {visible(name)}" for name in names]
    lines += [f"deny rule {visible(rule.text)} (applies either way)" for rule in deny]
    return TrustAsk(
        kind,
        lines,
        note,
        title="Trust the global config again?" + (f" ({', '.join(parts)})" if parts else ""),
        header=header,
        footer=footer,
        decline="Don't trust (keep them off; the next launch asks again)",
        accept="Trust the global config",
    )


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'s' if n != 1 else ''}"


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
