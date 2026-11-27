"""Headless mode never asks: whatever would open a dialog in the app is refused."""

from __future__ import annotations

from core.engine.approval import ApprovalRequest, Choice, Decision

HEADLESS_DENY = "no approval in headless mode (use --allow)"


class HeadlessApprover:
    """An `Approver` that denies every call; allow rules (settings, `--allow`) are decided before it is asked."""

    async def ask(self, request: ApprovalRequest) -> Decision:
        return Decision(Choice.DENY, HEADLESS_DENY)
