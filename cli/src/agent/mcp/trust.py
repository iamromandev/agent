"""Which project `mcp.json` files the user trusted: per folder, tied to the file's exact content."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def trust_store_path(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_DATA_HOME/agent/mcp-trust.json`, else `~/.local/share/agent/mcp-trust.json`.

    Only the process environment decides this, never `AGENT_SESSIONS_DIR` or any setting: settings can come
    from a `.env` in the workspace, and a repository must not be able to say where its own trust is kept.
    A relative `XDG_DATA_HOME` is ignored, as the XDG spec says.
    """
    data_home = env.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home and Path(data_home).is_absolute() else Path.home() / ".local" / "share"
    return base / "agent" / "mcp-trust.json"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class TrustStore:
    """`{folder: sha256 of the trusted file's bytes}` in one JSON file. A file that cannot be read trusts nothing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_bytes().decode("utf-8"))
        except (OSError, ValueError):  # missing, unreadable, not text, not JSON
            return {}
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            return {}
        return data

    def inside(self, workspace: Path) -> bool:
        """Is the store in the workspace (symlinks followed)? A repository could ship such a file: it is refused."""
        return self.path.resolve().is_relative_to(workspace.resolve())

    def is_trusted(self, workspace: Path, content: bytes) -> bool:
        if self.inside(workspace):
            return False
        return self._read().get(str(workspace)) == _digest(content)

    def trust(self, workspace: Path, content: bytes) -> None:
        """Remember `content` for `workspace`. Atomic: the old file stays whole if this fails (raises OSError)."""
        if self.inside(workspace):
            raise OSError(f"the trust store {self.path} is inside the workspace")
        data = self._read()  # a corrupt file reads as empty, so it is overwritten here
        data[str(workspace)] = _digest(content)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".mcp-trust-", suffix=".tmp")  # mode 0600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
