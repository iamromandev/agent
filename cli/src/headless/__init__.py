"""Headless mode: one agent turn without the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.headless.approver import HEADLESS_DENY as HEADLESS_DENY
from src.headless.approver import HeadlessApprover as HeadlessApprover
from src.runtime import NEW, StartChoice

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INT, EXIT_TERM = 0, 1, 2, 130, 143
STDIN_LIMIT = 1 << 20


class UsageError(Exception):
    """A bad invocation: the message goes to stderr and the exit code is 2."""


@dataclass(frozen=True)
class HeadlessOptions:
    prompt: str | None  # None: read stdin
    allow: tuple[str, ...] = ()
    quiet: bool = False
    trust_project: bool = False
    model: str | None = None
    start: StartChoice = NEW  # new, continue or resume (never pick)
    output: Literal["text", "json", "stream-json"] = "text"

