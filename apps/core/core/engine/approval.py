from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from core.engine.mode import Mode
from core.engine.tools.base import Preview


class Choice(StrEnum):
    ONCE = "once"
    ALWAYS = "always"
    PROJECT = "project"  # "Always here": also saved as a rule in the project settings file
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
    save_rule: str | None = None  # "Always here" would save this rule to the project settings file; None: not offered
    protected: bool = False  # a read of a protected path (guardrails): only Allow once / Deny, never Always
    protected_path: str = ""  # the resolved path, shown by the dialog (unescaped: the UI escapes it)
    subagent: str = ""  # the custom agent that made the call, "" for the main agent (unescaped: the UI escapes it)

    @property
    def allow_save(self) -> bool:
        return self.save_rule is not None


class Approver(Protocol):
    async def ask(self, request: ApprovalRequest) -> Decision: ...


class Grants:
    """What the user allowed with "Always". Per launch, in memory."""

    def __init__(self) -> None:
        self.tools: set[str] = set()
        self.exact: set[tuple[str, str]] = set()  # (tool, value of its key argument): granted per exact value

    @staticmethod
    def _key(tool: str, key: str | None) -> str | None:
        return key or ("command" if tool == "bash" else None)  # bash is granted per exact command, never per tool

    def covers(self, tool: str, args: dict[str, Any], mode: Mode, key: str | None = None) -> bool:
        """`key` names the argument a grant is tied to (`Tool.grant_arg`); without one the whole tool is granted."""
        if mode is Mode.PLAN:
            return False
        key = self._key(tool, key)
        if key is not None:
            value = args.get(key)
            return isinstance(value, str) and (tool, value) in self.exact
        return tool in self.tools

    def add(self, tool: str, args: dict[str, Any], key: str | None = None) -> None:
        key = self._key(tool, key)
        if key is None:
            self.tools.add(tool)
            return
        value = args.get(key)
        if isinstance(value, str):
            self.exact.add((tool, value))

    def revoke(self, tool: str) -> None:
        """Forget every grant for `tool`, whatever it was keyed by."""
        self.tools.discard(tool)
        self.exact = {(name, value) for name, value in self.exact if name != tool}
