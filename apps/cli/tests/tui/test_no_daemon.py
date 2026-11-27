"""Plain `agent` runs its session in-process: no daemon socket, pid file or log (AC2)."""

from __future__ import annotations

import os
from pathlib import Path

from core.engine.provider.base import Done, TextDelta

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send


async def test_a_turn_in_the_app_creates_nothing_in_the_daemon_run_folder(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "hi")
        await settled(app)
        assert len(scripted(app).requests) == 1
    assert not (Path(os.environ["XDG_DATA_HOME"]) / "agent" / "run").exists()
