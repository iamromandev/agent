"""`agent --shared`: the app on a `RemotePort` to a daemon running in the test (AC5, AC7, AC12, AC14)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from core.client.daemon import DaemonClient
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.daemon import data_home, serve
from core.testing.fakes import Gate
from src.tui.app import AgentApp
from src.tui.remote import LOST, NOT_SHARED, RemotePort
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, UserEntry

from tests.tui.conftest import type_and_send, until

__all__ = ["data_home", "serve"]

REPLY = [TextDelta("ok"), Done(1, 1)]
BASH = [ToolCallEvent(ToolCall("b1", "bash", {"command": "true"})), Done(1, 1)]


def notices(app: AgentApp) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


async def _open(app: AgentApp, pilot) -> str:
    port = app.port
    assert isinstance(port, RemotePort)
    await until(pilot, lambda: bool(port.session_id) and not port.switching)
    return port.session_id


async def test_a_dialog_answered_in_another_client_closes_with_a_notice(serve) -> None:
    served = await serve([BASH, REPLY])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    other = DaemonClient(served.socket, "B")
    try:
        async with app.run_test() as pilot:
            session_id = await _open(app, pilot)
            await type_and_send(pilot, "run it")
            await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            (ask,) = served.daemon.host(session_id).asks.pending()
            assert await other.answer(ask.ask_id, {"choice": "once"}) == "ok"
            await until(pilot, lambda: not isinstance(app.screen, ApprovalScreen))
            await until(pilot, lambda: any("answered in another client" in t for t in notices(app)))
            await until(pilot, lambda: [e.text for e in app.query(AssistantEntry)] == ["ok"])
            assert [e.text for e in app.query(UserEntry)] == ["run it"]
    finally:
        await other.aclose()


def _global_lsp(ws: Path) -> None:
    folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    servers = {"mine": {"command": "/bin/sh", "args": ["-c", "x"], "extensions": [".m"]}}
    (folder / "settings.json").write_text(json.dumps({"lsp": {"servers": servers}}))


async def test_lsp_in_shared_mode_shows_the_daemons_servers(serve) -> None:
    served = await serve([REPLY], setup=_global_lsp)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await type_and_send(pilot, "/lsp")
        await until(pilot, lambda: any("mine (global) idle" in t for t in notices(app)))


async def test_esc_interrupts_the_daemons_turn(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY]])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        session_id = await _open(app, pilot)
        await type_and_send(pilot, "slow")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await pilot.press("escape")
        host = served.daemon.host(session_id)
        await until(pilot, lambda: not host.turn_active)
        await until(pilot, lambda: any("interrupted" in t for t in notices(app)))


async def test_terminals_workflows_images_and_custom_commands_are_refused(serve) -> None:
    served = await serve([REPLY])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await pilot.press("ctrl+t")
        for command in ("/workflow", "/images"):
            await type_and_send(pilot, command)
        await until(pilot, lambda: sum(NOT_SHARED in t for t in notices(app)) == 3)
        await type_and_send(pilot, "hi")
        await until(pilot, lambda: bool(served.provider.requests))
    offered = set(served.provider.requests[0][1])
    assert not {name for name in offered if name.startswith("term_")}
    assert not offered & {"generate_image", "edit_image", "upscale_image", "speak", "assemble_video"}


async def test_a_lost_daemon_keeps_the_typed_text(serve) -> None:
    served = await serve()
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await pilot.press(*"half typed")
        served.daemon.request_stop()
        await until(pilot, lambda: any(LOST in t for t in notices(app)), timeout=10)
        assert app.query_one(ChatInput).text == "half typed"


async def test_a_custom_command_and_an_image_paste_are_refused(serve) -> None:
    served = await serve()
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await type_and_send(pilot, "/mycommand")
        app.query_one(ChatInput).post_message(ChatInput.PasteImage())
        await until(pilot, lambda: sum(NOT_SHARED in t for t in notices(app)) == 2)
        assert any("custom commands" in t for t in notices(app))
        assert any("pasting images" in t for t in notices(app))


async def test_a_message_queued_by_another_client_shows_as_queued_1(serve) -> None:
    from src.tui.widgets.queue_panel import QueuePanel

    gate = Gate()
    served = await serve([[gate, *REPLY]])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    other = DaemonClient(served.socket, "B")
    try:
        async with app.run_test() as pilot:
            session_id = await _open(app, pilot)
            await type_and_send(pilot, "slow")
            await asyncio.wait_for(gate.reached.wait(), 5)
            assert (await other.send(session_id, "later"))["status"] == "queued"
            await until(pilot, lambda: app.query_one(QueuePanel).border_title == " queued 1 ")
            gate.release.set()
    finally:
        await other.aclose()


async def test_opening_a_live_session_shows_its_turn_queue_and_pending_approval(serve) -> None:
    from src.tui.widgets.queue_panel import QueuePanel
    from src.tui.widgets.status import StatusBar

    served = await serve([BASH, REPLY])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    other = DaemonClient(served.socket, "B")
    try:
        async with app.run_test() as pilot:
            await _open(app, pilot)
            live = (await other.open_session(None))["session_id"]
            await other.send(live, "run it")
            host = served.daemon.host(live)
            await until(pilot, lambda: bool(host.asks.pending()))
            assert (await other.send(live, "after"))["status"] == "queued"
            port = app.port
            assert isinstance(port, RemotePort)
            (info,) = [i for i in await port.list_sessions() if i.id == live]
            assert await app._show_session(info)
            await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            assert app.query_one(StatusBar).busy
            assert app.query_one(QueuePanel).border_title == " queued 1 "
    finally:
        await other.aclose()


async def test_a_reconnect_snapshot_closes_dialogs_whose_ask_is_gone(serve) -> None:
    from core.wire import models as wm
    from src.tui.remote import Reconnected

    served = await serve([BASH, REPLY])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await type_and_send(pilot, "run it")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        port = app.port
        assert isinstance(port, RemotePort)
        port._asks.clear()  # answered elsewhere while this client was away
        await app._on_port_event(Reconnected(wm.Snapshot(session_id=port.session_id, seq=port._last_seq)))
        await until(pilot, lambda: not isinstance(app.screen, ApprovalScreen))


async def test_an_ask_that_arrives_while_quitting_is_left_for_the_other_clients(serve) -> None:
    from core.engine.approval import ApprovalRequest
    from core.host.asks import AskEvent

    served = await serve()
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    answered: list[str] = []
    async with app.run_test() as pilot:
        await _open(app, pilot)
        app.port.answer = lambda ask_id, answer: answered.append(ask_id) or "ok"  # ty: ignore[invalid-assignment]
        app._shutting_down = True
        app._on_ask(AskEvent("ASK1", "approval", ApprovalRequest("bash", "bash", None, True)))
        app._shutting_down = False
    assert answered == []


async def test_an_undo_in_another_client_drops_the_turn_here(serve) -> None:
    served = await serve([REPLY])
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    other = DaemonClient(served.socket, "B")
    try:
        async with app.run_test() as pilot:
            session_id = await _open(app, pilot)
            await type_and_send(pilot, "hi")
            await until(pilot, lambda: [e.text for e in app.query(AssistantEntry)] == ["ok"])
            host = served.daemon.host(session_id)
            await until(pilot, lambda: not host.turn_active)
            assert (await other.undo(session_id))["prompt"] == "hi"
            await until(pilot, lambda: any("undone in another client" in t for t in notices(app)))
            await until(pilot, lambda: not list(app.query(UserEntry)))
    finally:
        await other.aclose()


@pytest.mark.parametrize("choice", ["wait", "quit"])
async def test_a_daemon_that_does_not_come_back_asks_keep_waiting_or_quit(serve, monkeypatch, choice: str) -> None:
    import src.tui.remote as remote
    from src.tui.screens.daemon_gone import DaemonGoneScreen
    from textual.widgets import OptionList

    monkeypatch.setattr(remote, "RECONNECT_WAIT", 0.3)
    monkeypatch.setattr(remote, "RETRY_STEP", 0.05)
    served = await serve()
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    exits: list[object] = []
    async with app.run_test() as pilot:
        await _open(app, pilot)
        monkeypatch.setattr(app, "exit", lambda *a, **k: exits.append(k))
        await pilot.press(*"kept")
        served.daemon.request_stop()
        await asyncio.wait_for(served.task, 10)
        await until(pilot, lambda: isinstance(app.screen, DaemonGoneScreen), timeout=10)
        options = app.screen.query_one(OptionList)
        options.highlighted = options.get_option_index(choice)
        await pilot.press("enter")
        await until(pilot, lambda: not isinstance(app.screen, DaemonGoneScreen))
        if choice == "quit":
            assert exits
        else:
            assert not exits
            await serve()  # it comes back
            await until(pilot, lambda: any("reconnected to the agent daemon" in t for t in notices(app)), timeout=10)
        assert app.query_one(ChatInput).text == "kept"


async def test_a_daemon_that_cannot_be_reached_at_start_asks_and_quit_names_why(tmp_path, monkeypatch) -> None:
    from src.tui.screens.daemon_gone import DaemonGoneScreen
    from textual.widgets import OptionList

    app = AgentApp(RemotePort(tmp_path / "none.sock", tmp_path, client_id="A"))
    exits: list[dict] = []
    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "exit", lambda *a, **k: exits.append(k))
        await until(pilot, lambda: isinstance(app.screen, DaemonGoneScreen))
        options = app.screen.query_one(OptionList)
        options.highlighted = options.get_option_index("quit")
        await pilot.press("enter")
        await until(pilot, lambda: bool(exits))
    assert exits[0]["return_code"] == 1 and "cannot be reached" in exits[0]["message"]
