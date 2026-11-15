"""AC4: the approver approves an allowed call, denies the rest with a note, and records both."""

from __future__ import annotations

from evals.approver import AutoApprover
from evals.record import RunRecord
from src.agent.approval import ApprovalRequest, Choice


def _request(tool: str, label: str) -> ApprovalRequest:
    return ApprovalRequest(tool, label, None, True)


async def test_allowed_call_is_approved_once_and_recorded() -> None:
    record = RunRecord("t", 1)
    approver = AutoApprover(lambda request: request.tool == "write", record)
    decision = await approver.ask(_request("write", "write a.py"))
    assert decision.choice is Choice.ONCE
    assert record.approvals == [{"tool": "write", "label": "write a.py", "allowed": True}]


async def test_other_call_is_denied_with_a_note_and_recorded() -> None:
    record = RunRecord("t", 1)
    approver = AutoApprover(lambda request: request.tool == "write", record)
    decision = await approver.ask(_request("bash", "bash rm -rf build"))
    assert decision.choice is Choice.DENY
    assert decision.note == "not allowed in this eval"
    assert record.approvals == [{"tool": "bash", "label": "bash rm -rf build", "allowed": False}]
