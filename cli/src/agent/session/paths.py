"""Where sessions live on disk: <root>/<workspace id>/<ulid>.jsonl."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path


def sessions_root(configured: str | None, env: Mapping[str, str] = os.environ) -> Path:
    """The folder holding every workspace's sessions. Always absolute.

    A relative `configured` path is made absolute against the current folder at this call, so call it once
    at startup: a later `chdir` must not move the sessions. A relative `XDG_DATA_HOME` is ignored, as the
    XDG spec says.
    """
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else Path.cwd() / path
    data_home = env.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home and Path(data_home).is_absolute() else Path.home() / ".local" / "share"
    return base / "agent" / "sessions"


def workspace_id(workspace: Path) -> str:
    return hashlib.sha256(os.fsencode(str(workspace.resolve()))).hexdigest()[:12]


def workspace_dir(root: Path, workspace: Path) -> Path:
    return root / workspace_id(workspace)
