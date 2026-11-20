from __future__ import annotations

from src.engine.mode import Mode


def test_tab_cycle_is_build_edits_plan() -> None:
    assert Mode.BUILD.next() is Mode.EDITS
    assert Mode.EDITS.next() is Mode.PLAN
    assert Mode.PLAN.next() is Mode.BUILD
