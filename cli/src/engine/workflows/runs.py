"""An index of the runs of one workspace, kept in the agent's data folder (never in the workspace)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.mcp.trust import trust_store_path

MAX_RUNS = 50


@dataclass(frozen=True)
class RunEntry:
    run: str
    name: str
    folder: str  # relative to the workspace
    state: str  # "running", "failed" or "done"


class RunIndex:
    """`{"runs": [...]}`, oldest first, at most `MAX_RUNS`. A file that cannot be read is an empty index."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_workspace(cls, workspace: Path, env: Mapping[str, str] = os.environ) -> RunIndex:
        """`<data>/agent/workflows/<sha256(workspace)[:16]>.json`, the data folder the trust store uses."""
        key = hashlib.sha256(os.fsencode(str(workspace))).hexdigest()[:16]
        return cls(trust_store_path(env, f"workflows/{key}.json"))

    def _read(self) -> list[RunEntry]:
        try:
            data = json.loads(self.path.read_bytes().decode("utf-8"))
            return [RunEntry(e["run"], e["name"], e["folder"], e["state"]) for e in data["runs"]]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _write(self, entries: list[RunEntry]) -> None:
        text = json.dumps({"runs": [vars(e) for e in entries]}, indent=1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".runs-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp)
            raise

    def record(self, run: str, name: str, folder: str, state: str) -> None:
        """Add the run (or update it) as the newest entry."""
        entries = [e for e in self._read() if (e.run, e.folder) != (run, folder)]  # the same id may exist per workflow
        entries.append(RunEntry(run, name, folder, state))
        self._write(entries[-MAX_RUNS:])

    def latest_unfinished(self, usable: Callable[[RunEntry], bool] = lambda entry: True) -> RunEntry | None:
        """The newest run that is not done and that `usable` accepts (a run whose folder was undone is skipped)."""
        return next((e for e in reversed(self._read()) if e.state != "done" and usable(e)), None)

    def find(self, run: str) -> RunEntry | None:
        """The run with this id (the newest when several workflows share it), or the only one whose id starts with it."""
        entries = self._read()
        exact = [e for e in entries if e.run == run]
        if exact:
            return exact[-1]
        starts = [e for e in entries if run and e.run.startswith(run)]
        return starts[0] if len(starts) == 1 else None
