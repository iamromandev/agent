"""The agent's own folders, which accept-edits mode never writes into."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def _xdg(env: Mapping[str, str], name: str, fallback: Path) -> Path:
    value = env.get(name)
    return Path(value) if value and Path(value).is_absolute() else fallback


def agent_folders(env: Mapping[str, str], home: Path, sessions: Path | None = None) -> tuple[Path, ...]:
    """The agent's own folders: `<config>/agent` (settings, MCP servers, commands, skills), `<data>/agent` (trust
    stores, checkpoints) and the sessions folder when given. Accept-edits mode never approves a write into them."""
    config = _xdg(env, "XDG_CONFIG_HOME", home / ".config") / "agent"
    data = _xdg(env, "XDG_DATA_HOME", home / ".local" / "share") / "agent"
    return (config, data) if sessions is None else (config, data, sessions)
