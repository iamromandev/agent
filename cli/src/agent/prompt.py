from __future__ import annotations

import codecs
import os
import stat
from datetime import date
from pathlib import Path

from src.agent.mode import Mode

AGENTS_MD_LIMIT = 32 * 1024

_ROLE = (
    "You are a coding agent running in the user's terminal. You help with software tasks "
    "in the workspace by reading, searching and, when allowed, changing files and running commands."
)

_GUIDANCE = (
    "Use the tools to look before you answer. Read a file before you edit it. "
    "Prefer edit over write for existing files, and keep edits small. "
    "Use glob and grep to find files instead of guessing paths. "
    "Tool calls may be denied by the user; if one is, follow their note and do not retry it unchanged. "
    "Be concise."
)

_PLAN = "You are in PLAN mode: do not modify files; write/edit are unavailable. Explore, then propose a plan."


def _read_head(path: Path, limit: int) -> bytes | None:
    """Up to `limit` bytes of a regular file; None for anything else (FIFO, device, directory)."""
    try:
        # O_NONBLOCK so opening a FIFO returns at once; the fstat then rejects it (no TOCTOU).
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:  # missing, unreadable
        return None
    with os.fdopen(fd, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            return None
        return file.read(limit)


def _agents_md(workspace: Path) -> str | None:
    try:
        raw = _read_head(workspace / "AGENTS.md", AGENTS_MD_LIMIT + 1)
    except OSError:
        return None
    if raw is None:
        return None
    truncated = len(raw) > AGENTS_MD_LIMIT
    if truncated:
        raw = raw[:AGENTS_MD_LIMIT]
    # Not final when cut: a character split by the limit is dropped, while invalid bytes before
    # it become U+FFFD instead of vanishing.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(raw, final=not truncated)
    return text + ("\n[AGENTS.md truncated at 32 KB]" if truncated else "")


def build_system_prompt(workspace: Path, mode: Mode, today: date, os_name: str) -> str:
    parts = [
        _ROLE,
        f"Workspace: {workspace}\nMode: {mode.value}\nOS: {os_name}\nDate: {today.isoformat()}",
        _GUIDANCE,
    ]
    if mode is Mode.PLAN:
        parts.append(_PLAN)
    agents = _agents_md(workspace)
    if agents is not None:
        parts.append(f"## AGENTS.md\n{agents}")
    return "\n\n".join(parts)
