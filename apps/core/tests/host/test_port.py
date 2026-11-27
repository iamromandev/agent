"""The local port's pump: a failing handler must neither stop it nor leave an ask unanswered."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.engine import Agent
from core.engine.approval import ApprovalRequest, Approver, Choice, Decision
from core.engine.events import Notice
from core.engine.mentions_index import FileIndex
from core.engine.tools.base import ToolContext
from core.host.asks import AskEvent
from core.host.port import LocalPort
from core.host.session import SessionHost
from core.testing.fakes import ScriptedProvider

REQUEST = ApprovalRequest("bash", "bash ls", None, allow_always=True)


def _port(tmp_path: Path) -> LocalPort:
    root = tmp_path.resolve()
    provider = ScriptedProvider([])

    def factory(approver: Approver) -> Agent:
        return Agent(provider, [], approver, ToolContext(root, 5.0), 5)

    host = SessionHost(factory, provider, root, window_cap=8192)
    return LocalPort(host, FileIndex(root, host.guard))


async def test_an_ask_whose_dialog_cannot_be_shown_is_denied_and_reported(tmp_path: Path) -> None:
    port = _port(tmp_path)
    seen: list[object] = []

    async def on_event(event: object) -> None:
        seen.append(event)

    def on_ask(event: AskEvent) -> None:
        raise RuntimeError("no screen \x1b[31m" + "x" * 300)

    port.subscribe(on_event, on_ask)
    try:
        answer = await asyncio.wait_for(port.host.asks.ask("approval", REQUEST), 5)
        await asyncio.wait_for(port.host.bus.drained(), 5)
    finally:
        await port.close()

    assert answer == Decision(Choice.DENY, "could not show the dialog")
    (notice,) = [e for e in seen if isinstance(e, Notice)]
    assert notice.level == "error" and notice.text.startswith("Could not show the approval dialog: no screen \\x1b[31m")
    assert len(notice.text) <= len("Could not show the approval dialog: ") + 200


async def test_a_failing_event_handler_is_reported_and_the_pump_goes_on(tmp_path: Path) -> None:
    port = _port(tmp_path)
    seen: list[object] = []

    async def on_event(event: object) -> None:
        if event == "bad":
            raise ValueError("boom")
        seen.append(event)

    port.subscribe(on_event, lambda event: None)
    try:
        port.host.bus.publish("bad")
        port.host.bus.publish("good")
        await asyncio.wait_for(port.host.bus.drained(), 5)
    finally:
        await port.close()

    assert seen == ["good", Notice("error", "Display error: boom")]  # the notice is published after what was queued
