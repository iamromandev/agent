from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from src.agent.terminal.manager import TerminalManager
from src.agent.tools.base import ToolContext

from tests.helpers import assert_all_gone


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


@pytest.fixture
async def manager(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TerminalManager]:
    """A manager with a short grace period; every terminal it started is stopped and checked gone."""
    monkeypatch.setenv("SHELL", "/bin/sh")
    terminals = TerminalManager(workspace, grace=0.3)
    pids: list[int] = []

    def started(name: str, event: str) -> None:
        if event == "started" and (terminal := terminals.get(name)) is not None:
            pids.append(terminal.process.pid)

    terminals.listen(started)
    yield terminals
    await terminals.close_all()
    await assert_all_gone(*pids)
