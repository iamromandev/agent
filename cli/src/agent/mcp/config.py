"""Reading `mcp.json`: which MCP servers to start. A bad file or entry becomes a warning, never an error."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

Source = Literal["global", "project"]


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    source: Source = "global"
    overrides_global: bool = False  # a project server that replaced a global one of the same name


@dataclass(frozen=True)
class LoadedConfig:
    servers: list[ServerConfig]
    warnings: list[str]
    raw: bytes | None = None  # the exact bytes that were parsed, when the file was read (what trust is tied to)


def global_config_path(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/mcp.json`, else `~/.config/agent/mcp.json`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "mcp.json"


def project_config_path(workspace: Path) -> Path:
    """`<workspace>/.agent/mcp.json`."""
    return workspace / ".agent" / "mcp.json"


def _is_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _entry(name: str, raw: object, source: Source) -> ServerConfig | str:
    """The server for one entry, or what is wrong with it."""
    if not isinstance(raw, dict):
        return "must be an object"
    command: Any = raw.get("command")
    if "command" not in raw:
        return 'missing "command"'
    if not isinstance(command, str):
        return '"command" must be a string'
    if not command.strip():
        return '"command" must not be empty'
    args: Any = raw.get("args", [])
    if not _is_strings(args):
        return '"args" must be a list of strings'
    env: Any = raw.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        return '"env" must be an object of strings'
    return ServerConfig(name, command, tuple(args), dict(env), source)


def load_file(path: Path, source: Source) -> LoadedConfig:
    """Read one config file. A missing file is no servers and no warning; anything else wrong is a warning."""
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except FileNotFoundError:
        return LoadedConfig([], [])
    except (OSError, UnicodeDecodeError) as exc:
        return LoadedConfig([], [f"{path}: cannot read: {exc}"])
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return LoadedConfig([], [f"{path}: invalid JSON: {exc}"], raw)
    if not isinstance(data, dict):
        return LoadedConfig([], [f"{path}: top level must be an object"], raw)
    entries = data.get("servers")
    if not isinstance(entries, dict):
        return LoadedConfig([], [f'{path}: "servers" must be an object'], raw)
    servers: list[ServerConfig] = []
    warnings: list[str] = []
    for name, value in entries.items():
        entry = _entry(name, value, source)
        if isinstance(entry, str):
            warnings.append(f'{path}: server "{name}": {entry}')
        else:
            servers.append(entry)
    return LoadedConfig(servers, warnings, raw)


def merge(global_: LoadedConfig, project: LoadedConfig) -> LoadedConfig:
    """Both files' servers; a project server replaces the global one with the same name (in its place)."""
    overrides = {s.name: s for s in project.servers}
    servers = [replace(overrides.pop(s.name), overrides_global=True) if s.name in overrides else s for s in global_.servers]
    servers.extend(overrides.values())
    return LoadedConfig(servers, [*global_.warnings, *project.warnings])
