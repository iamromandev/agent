"""How long workflow calls took on this machine, kept in the agent's data folder (never the workspace)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from src.engine.fileio import read_regular
from src.engine.mcp.trust import trust_store_path

TIMINGS_FILE = "workflow-timings.json"
MAX_RECORDS = 20
MAX_BYTES = 1 << 20


def _number(value: object) -> float | None:
    """The value as a finite positive float, or None (a bool, a string, an integer too big for a float, ...)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except (OverflowError, ArithmeticError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


class TimingStore:
    """`{"timings": 1, "keys": {"<key>": [value, ...]}}`, the last `MAX_RECORDS` values per key, newest last.

    A file that is missing, unreadable, too big, not JSON or of another shape holds nothing. With `path` None the
    values stay in memory and nothing is written."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._memory: dict[str, list[float]] = {}

    @classmethod
    def default(cls, env: Mapping[str, str] = os.environ) -> TimingStore:
        return cls(trust_store_path(env, TIMINGS_FILE))

    def _read(self) -> dict[str, list[float]]:
        if self.path is None:
            return {key: list(values) for key, values in self._memory.items()}
        try:
            data = json.loads(read_regular(self.path, MAX_BYTES, follow=False).decode("utf-8"))
        except (OSError, ValueError, RecursionError):
            return {}
        keys = data.get("keys") if isinstance(data, dict) and data.get("timings") == 1 else None
        if not isinstance(keys, dict):
            return {}
        found: dict[str, list[float]] = {}
        for key, values in keys.items():
            if isinstance(key, str) and isinstance(values, list):
                good = [n for n in map(_number, values) if n is not None]
                found[key] = good[-MAX_RECORDS:]
        return found

    def values(self, key: str) -> list[float]:
        return self._read().get(key, [])

    def add(self, key: str, value: float) -> None:
        """Append a value (ignored unless finite and positive), keep the last `MAX_RECORDS`, write atomically.
        Raises `OSError` when the file cannot be written; the file's contents never make it raise. Read, change,
        write: two app instances adding at once can lose a record, but the atomic rename means the file is never
        corrupted."""
        number = _number(value)
        if number is None:
            return
        data = self._read()
        data[key] = [*data.get(key, []), number][-MAX_RECORDS:]
        if self.path is None:
            self._memory = data
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".workflow-timings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"timings": 1, "keys": data}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
