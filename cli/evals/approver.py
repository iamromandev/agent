"""The approver of a headless run: decides by the task's policy, never asks anyone."""

from __future__ import annotations

from collections.abc import Callable

from src.agent.approval import ApprovalRequest, Choice, Decision

from evals.record import RunRecord

DENIED = "not allowed in this eval"


class AutoApprover:
    def __init__(self, allow: Callable[[ApprovalRequest], bool], record: RunRecord) -> None:
        self._allow = allow
        self._record = record

    async def ask(self, request: ApprovalRequest) -> Decision:
        allowed = self._allow(request)
        self._record.approvals.append({"tool": request.tool, "label": request.label, "allowed": allowed})
        return Decision(Choice.ONCE) if allowed else Decision(Choice.DENY, DENIED)
