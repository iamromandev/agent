"""The text of the listing commands (`/skills`, `/agents`, `/permissions`, `/hooks`, `/mcp`, `/guardrails`), UI-free so
the app and the daemon's `info` route show the same lines. Everything that came from a file is escaped."""

from __future__ import annotations

from core.engine.agents.loader import READ_ONLY_TOOLS, Agents
from core.engine.configguard import ConfigGuard
from core.engine.guardrails.paths import (
    BUILTIN_HOME,
    BUILTIN_NAMES,
    MAX_TRACKED_DEPTH,
    MAX_TRACKED_DIRS,
    MAX_TRACKED_FILES,
)
from core.engine.guardrails.setup import Guardrails
from core.engine.hooks.setup import Hooks
from core.engine.lsp.hub import LspHub
from core.engine.mcp.hub import McpHub
from core.engine.mcp.names import tool_name
from core.engine.permissions.setup import Rules
from core.engine.skills.loader import Skills
from core.engine.text import escape, escape_display

__all__ = [
    "agents_summary",
    "guardrails_summary",
    "hooks_summary",
    "lsp_summary",
    "mcp_summary",
    "permissions_summary",
    "skills_summary",
]


def skills_summary(skills: Skills) -> str:
    """`<name> (<source>): <description>` per skill, then the launch warnings (file text, so escaped)."""
    lines = [f"{escape(s.name)} ({s.source}): {escape_display(' '.join(s.description.split()))}" for s in skills.sorted()]
    lines.extend(skills.warnings)
    return "\n".join(lines) if lines else "No skills."


def agents_summary(agents: Agents) -> str:
    """Per agent `<name> (<source>)[ model <model>]: <description>` and its tools, then the launch warnings."""
    lines: list[str] = []
    for a in agents.sorted():
        shown = a.resolved_model or a.model
        model = f" model {escape_display(shown)}" if shown else ""
        if a.missing:
            model += " (not installed)"
        lines.append(f"{escape(a.name)} ({a.source}){model}: {escape_display(' '.join(a.description.split()))}")
        tools = READ_ONLY_TOOLS if a.tools is None else a.tools
        lines.append(f"  tools: {', '.join(tools) or 'none'}")
    lines.extend(agents.warnings)
    return "\n".join(lines) if lines else "No agents."


def permissions_summary(rules: Rules) -> str:
    lines = rules.listing()
    return "\n".join(lines) if lines else "no permission rules"


def hooks_summary(hooks: Hooks) -> str:
    lines = hooks.listing()
    return "\n".join(lines) if lines else "No hooks."


def guardrails_summary(guard: Guardrails, config_guard: ConfigGuard | None) -> str:
    """What the guardrails do in this run. Names and paths only, never a value; all of it escaped."""
    lines: list[str] = []
    # `.env` and `.env.*` are shown as one `.env*` (matching is unchanged: the template files stay unprotected)
    names = [".env*" if name == ".env" else name for name in BUILTIN_NAMES if name != ".env.*"]
    builtin = [pattern.removesuffix("/**") for pattern in BUILTIN_HOME] + names
    extra = guard.paths.extra_count
    lines.append("Protected: " + ", ".join(builtin) + (f" (+{extra} from settings)" if extra else ""))
    homes = guard.paths.home_folders
    if len(homes) > 1:  # `$HOME` and the account's recorded home differ: `~` in a pattern means both
        lines.append("Home folders: " + ", ".join(escape(str(home)) for home in homes))
    if guard.paths.tracking_truncated:
        lines.append(
            f"Hard links to protected files are tracked for the first {MAX_TRACKED_FILES:,} files only "
            f"(and {MAX_TRACKED_DEPTH} folder levels, {MAX_TRACKED_DIRS:,} folders) under protected folders "
            "and, for protected names, in the workspace"
        )
    removed = ", ".join(escape(name) for name in guard.removed_env) or "none"
    lines.append(f"Removed from child environment: {removed}")
    lines.append(f"Redacted this run: {guard.redactions}")
    if config_guard is not None:  # the changes the config watch saw (escaped there) and the global marker
        lines.extend(config_guard.summary())
    return "\n".join(lines)


def mcp_summary(mcp: McpHub | None) -> str:
    """`<name> (<source>) <status>[: reason]` per server, the tool names of a ready one indented below it."""
    states = mcp.states() if mcp is not None else []
    if not states:
        return "No MCP servers configured (global: ~/.config/agent/mcp.json, project: .agent/mcp.json)"
    lines: list[str] = []
    for state in states:
        config = state.config
        line = f"{config.name} ({config.source}) {state.status}"
        line = f"{line}: {state.reason}" if state.reason else line
        if config.overrides_global:
            line += f" · overrides the global '{config.name}'"
        lines.append(line)
        if state.status == "ready":
            lines.extend(f"  {name}" for name in dict.fromkeys(tool_name(config.name, i.name) for i in state.tools))
    return "\n".join(lines)


def lsp_summary(lsp: LspHub | None) -> str:
    """`<name> (<source>) <state>[: reason]` per language server in table order, then the setup warnings."""
    if lsp is None:
        return 'LSP is off (set "lsp": true in settings)'
    lines = []
    for name, source, state, reason in lsp.listing():
        line = f"{escape_display(name)} ({source}) {state}"
        lines.append(f"{line}: {escape_display(reason)}" if reason else line)
    lines.extend(lsp.warnings)
    return "\n".join(lines) if lines else "No language servers found."
