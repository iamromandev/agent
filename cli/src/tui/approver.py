"""Approver that asks the user through a modal screen."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from src.agent.approval import ApprovalRequest, Choice, Decision
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.transcript import Transcript

if TYPE_CHECKING:
    from src.tui.app import AgentApp


class TuiApprover:
    def __init__(self, app: AgentApp) -> None:
        self.app = app

    async def ask(self, request: ApprovalRequest) -> Decision:
        """Runs inside the turn worker (push_screen_wait needs one)."""
        screen = ApprovalScreen(request)
        try:
            result = await self.app.push_screen_wait(screen)
        except asyncio.CancelledError:
            # The turn was cancelled with the dialog open (ctrl+c): do not leave it on screen.
            await self._close(screen)
            raise
        if result is None:  # Escape: deny, and stop the turn
            self.app.interrupt_turn()
            await asyncio.sleep(0)  # the cancel is delivered here, so it cannot slip past a fast provider
            return Decision(Choice.DENY, "interrupted")  # only reached if nothing was cancelled
        return result

    async def _close(self, screen: ApprovalScreen) -> None:
        """Take the dialog off the stack even when another screen sits above it."""
        try:
            while screen in (stack := self.app.screen_stack) and len(stack) > 1:
                if self.app.screen is screen:
                    screen.dismiss(None)
                else:
                    self.app.pop_screen()
        except Exception as exc:  # report it: a dialog that cannot be closed must not fail silently
            with contextlib.suppress(Exception):
                await self.app.query_one(Transcript).add_notice("error", f"Could not close the approval dialog: {exc}")
        if screen in self.app.screen_stack:
            with contextlib.suppress(Exception):
                await self.app.query_one(Transcript).add_notice("error", "The approval dialog is still open")
