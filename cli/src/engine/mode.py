from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    BUILD = "build"
    EDITS = "edits"
    PLAN = "plan"

    def next(self) -> Mode:
        """The mode Tab switches to: BUILD → EDITS → PLAN → BUILD."""
        order = (Mode.BUILD, Mode.EDITS, Mode.PLAN)
        return order[(order.index(self) + 1) % len(order)]
