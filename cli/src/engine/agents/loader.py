"""Finding and reading custom subagents. Files come from a repository: nothing in them is run."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.commands.loader import NAME, Source, front_matter
from src.engine.fileio import read_regular
from src.engine.permissions.setup import MAX_WARNINGS
from src.engine.provider.models import installed_name
from src.engine.text import escape, escape_display, one_line

MAX_BYTES = 64 * 1024
DESCRIPTION_CHARS = 200
SECTION_CHARS = 8_000
RESERVED_NAMES = frozenset({"default"})
READ_ONLY_TOOLS = ("read", "glob", "grep", "web_fetch", "web_search", "skill")  # today's `task` child, in order
AGENT_TOOLS = frozenset({*READ_ONLY_TOOLS, "write", "edit", "bash"})
_HEADER = (
    "Available agents (pass one as the task tool's agent argument when its description matches the job; "
    "omit agent for the built-in research agent):"
)


@dataclass(frozen=True)
class AgentDef:
    name: str
    description: str  # raw from the file, not escaped
    prompt: str  # the body, raw, without its leading blank lines
    tools: tuple[str, ...] | None  # None: no `tools:` key, the default read-only set
    model: str | None  # as written; None when absent or blank
    resolved_model: str | None  # set by `Agents.with_models`; None follows the parent's model
    source: Source
    path: Path


@dataclass(frozen=True)
class Agents:
    by_name: Mapping[str, AgentDef]
    warnings: tuple[str, ...]  # already escaped and capped

    @staticmethod
    def empty() -> Agents:
        return Agents({}, ())

    def get(self, name: str) -> AgentDef | None:
        return self.by_name.get(name)

    def sorted(self) -> list[AgentDef]:
        return [self.by_name[name] for name in sorted(self.by_name)]

    def names(self) -> list[str]:
        return sorted(self.by_name)

    def section(self) -> str | None:
        """The "Available agents" prompt text, at most `SECTION_CHARS` long; `None` when there are no agents."""
        agents = self.sorted()
        if not agents:
            return None
        lines = [f"- {a.name}: {one_line(a.description, DESCRIPTION_CHARS)}" for a in agents]
        if len(_HEADER) + sum(1 + len(line) for line in lines) <= SECTION_CHARS:
            return "\n".join([_HEADER, *lines])
        budget = SECTION_CHARS - len(f"\nand {len(lines)} more agents (see /agents)")  # the longest tail
        kept: list[str] = []
        total = len(_HEADER)
        for line in lines:
            if total + 1 + len(line) > budget:
                break
            kept.append(line)
            total += 1 + len(line)
        return "\n".join([_HEADER, *kept, f"and {len(lines) - len(kept)} more agents (see /agents)"])

    def needs_models(self) -> bool:
        return any(a.model for a in self.by_name.values())

    def with_models(self, installed: list[str] | None) -> Agents:
        """A copy with each `model:` checked against the installed names (`None`: unknown, taken as written)."""
        found: dict[str, AgentDef] = {}
        warnings = list(self.warnings)
        extra: list[str] = []
        for name, agent in self.by_name.items():
            if not agent.model:
                found[name] = agent
            elif installed is None:
                found[name] = dataclasses.replace(agent, resolved_model=agent.model)
            elif (match := installed_name(agent.model, installed)) is not None:
                found[name] = dataclasses.replace(agent, resolved_model=match)
            else:
                found[name] = agent
                extra.append(
                    escape(f"{agent.path}: model '{escape_display(agent.model)}' is not installed; using the session's model")
                )
        return Agents(found, _cap([*warnings, *extra], escaped=True))


def _cap(warnings: list[str], escaped: bool = False) -> tuple[str, ...]:
    shown = warnings if escaped else [escape(w) for w in warnings]
    if len(shown) > MAX_WARNINGS:
        # An earlier "and N more warnings" line, if any, is already counted as one of the shown ones.
        shown = [*shown[:MAX_WARNINGS], f"and {len(shown) - MAX_WARNINGS} more warnings"]
    return tuple(shown)


def global_agents_dir(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/agents`, else `~/.config/agent/agents`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "agents"


def project_agents_dir(workspace: Path) -> Path:
    return workspace / ".agent" / "agents"


def _parse_tools(value: str, path: Path, warnings: list[str]) -> tuple[str, ...] | None:
    """The listed tools; None (the default read-only set) when the list names none at all."""
    names: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return None
    tools: list[str] = []
    for name in names:
        if name not in AGENT_TOOLS:
            warnings.append(f"{path}: tool '{name}' is not allowed in an agent; ignored")
        else:
            tools.append(name)
    if not tools:
        warnings.append(f"{path}: no valid tools; the agent has none")
    return tuple(tools)


def _read(path: Path, name: str, source: Source, warnings: list[str]) -> AgentDef:
    data = read_regular(path, MAX_BYTES, follow=source == "global")
    keys, body = front_matter(data.decode("utf-8"))
    if keys is None:
        raise ValueError("no front matter")
    if not keys.get("description"):
        raise ValueError("description is missing or empty")
    prompt = body.lstrip("\n")
    if not prompt.strip():
        raise ValueError("the body is empty")
    tools = _parse_tools(keys["tools"], path, warnings) if "tools" in keys else None
    model: str | None = None
    if "model" in keys:
        if keys["model"]:
            model = keys["model"]
        else:
            warnings.append(f"{path}: model is empty; using the session's model")
    shown = path.resolve() if source == "global" else path.absolute()
    return AgentDef(name, keys["description"], prompt, tools, model, None, source, shown)


def _load_dir(directory: Path, source: Source, found: dict[str, AgentDef], warnings: list[str]) -> None:
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        warnings.append(f"{directory}: {exc.strerror or exc}")
        return
    for path in entries:
        if path.suffix != ".md":
            continue
        name = path.stem
        if not NAME.fullmatch(name):
            warnings.append(f"{path}: not a valid agent name")
        elif name in RESERVED_NAMES:
            warnings.append(f"{path}: '{name}' is reserved for the built-in agent; skipped")
        elif source == "project" and path.is_symlink():
            warnings.append(f"{path}: a symlink is not followed in the project folder")
        else:
            try:
                found[name] = _read(path, name, source, warnings)
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
                reason = exc.strerror or str(exc) if isinstance(exc, OSError) else str(exc)
                warnings.append(f"{path}: {reason}")


def load_agents(workspace: Path, env: Mapping[str, str] = os.environ) -> Agents:
    """Read both folders once. A project agent replaces a global one; problems become warnings."""
    found: dict[str, AgentDef] = {}
    warnings: list[str] = []
    _load_dir(global_agents_dir(env), "global", found, warnings)
    project = project_agents_dir(workspace)
    if project.parent.is_symlink() or project.is_symlink():
        warnings.append(f"{project}: a symlink is not followed in the project folder")
    else:
        _load_dir(project, "project", found, warnings)
    return Agents(found, _cap(warnings))
