from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from src.agent.mode import Mode
from src.agent.tools.base import Preview


class Choice(StrEnum):
    ONCE = "once"
    ALWAYS = "always"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    choice: Choice
    note: str = ""


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    label: str
    preview: Preview | None
    allow_always: bool  # False for bash in plan mode


class Approver(Protocol):
    async def ask(self, request: ApprovalRequest) -> Decision: ...


class Grants:
    """What the user allowed with "Always". Per launch, in memory."""

    def __init__(self) -> None:
        self.tools: set[str] = set()
        self.commands: set[str] = set()  # bash is granted per exact command, never per tool

    def covers(self, tool: str, args: dict[str, Any], mode: Mode) -> bool:
        if mode is Mode.PLAN:
            return False
        if tool == "bash":
            return args.get("command") in self.commands
        return tool in self.tools

    def add(self, tool: str, args: dict[str, Any]) -> None:
        if tool == "bash":
            command = args.get("command")
            if isinstance(command, str):
                self.commands.add(command)
        else:
            self.tools.add(tool)
