"""The sandbox for spawned processes. Phase 1 ships only the shape: nothing is wrapped yet."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sandbox:
    profile: str | None  # None: unsandboxed
    reason: str  # "on (macOS sandbox-exec)", "off: disabled in settings", ...
    network: bool
    writable: tuple[Path, ...]

    def wrap(self, argv: Sequence[str]) -> list[str]:
        """`argv` as it must be started: unchanged without a profile."""
        return list(argv)
