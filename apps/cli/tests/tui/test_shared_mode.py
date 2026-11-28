"""`agent --shared`: the app on a `RemotePort` to a daemon running in the test (AC5, AC7, AC12, AC14)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from core.client.daemon import DaemonClient
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import SessionInfo
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


async def _open_port(app: AgentApp, pilot) -> RemotePort:
    await _open(app, pilot)
    assert isinstance(app.port, RemotePort)
    return app.port


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


async def test_an_image_paste_is_refused(serve) -> None:
    served = await serve()
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        app.query_one(ChatInput).post_message(ChatInput.PasteImage())
        await until(pilot, lambda: any("pasting images" in t and NOT_SHARED in t for t in notices(app)))


def _review(ws: Path) -> None:
    folder = ws / ".agent" / "commands"
    folder.mkdir(parents=True)
    (folder / "review.md").write_text("---\ndescription: Look it over\n---\nReview $ARGUMENTS\n")


async def test_the_daemons_commands_are_suggested_listed_in_help_and_run_with_their_label(serve) -> None:
    served = await serve([REPLY], setup=_review)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        assert "/review  Look it over (project)" in app._help()
        await pilot.press(*"/rev")
        await until(pilot, lambda: any(name == "review" for name, *_ in app._suggestion_entries()))
        app.query_one(ChatInput).clear()
        await type_and_send(pilot, "/review x")
        await until(pilot, lambda: bool(served.provider.requests))
        await until(pilot, lambda: any(e.text == "Review x" for e in app.query(UserEntry)))
    assert served.provider.requests[0][0][-1].content == "Review x"


async def test_an_unknown_command_warns_and_starts_no_turn(serve) -> None:
    served = await serve([REPLY], setup=_review)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await type_and_send(pilot, "/nope")
        await until(pilot, lambda: any("Unknown command: /nope" in t for t in notices(app)))
    assert served.provider.requests == []


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
        # the screen is pushed before its options are mounted: wait for both
        await until(pilot, lambda: isinstance(app.screen, DaemonGoneScreen) and bool(app.screen.query(OptionList)), timeout=10)
        if choice == "wait":
            # back before the answer: with the 0.3 s wait, a daemon started after it could miss the next wait on a
            # loaded machine and the dialog would come up again, with nobody to answer it
            await serve()
        options = app.screen.query_one(OptionList)
        options.highlighted = options.get_option_index(choice)
        await pilot.press("enter")
        await until(pilot, lambda: not isinstance(app.screen, DaemonGoneScreen))
        if choice == "quit":
            assert exits
        else:
            assert not exits
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


def _slow_hook(pid_file: Path):
    def setup(ws: Path) -> None:
        folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
        folder.mkdir(parents=True, exist_ok=True)
        command = f"echo $$ >> {pid_file}; exec sleep 30"
        (folder / "settings.json").write_text(json.dumps({"hooks": {"SessionStart": [{"command": command, "timeout": 60}]}}))

    return setup


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_a_key_interrupts_the_daemons_session_start_hook_for_the_client_that_opened_the_session(
    serve, tmp_path, key: str
) -> None:
    pid_file = tmp_path / "pids"
    served = await serve(setup=_slow_hook(pid_file))
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        port = app.port
        assert isinstance(port, RemotePort)
        await until(pilot, lambda: pid_file.exists() and port.session_starting)
        await pilot.press(key)
        await until(pilot, lambda: any("hook SessionStart interrupted" in t for t in notices(app)))
        await until(pilot, lambda: not port.session_starting)
        assert app.is_running  # ctrl+c did not quit the client
    assert _gone(int(pid_file.read_text().split()[0]))


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_a_key_interrupts_the_hook_for_a_client_that_attached_while_it_ran(serve, tmp_path, key: str) -> None:
    pid_file = tmp_path / "pids"
    served = await serve(setup=_slow_hook(pid_file))
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    other = DaemonClient(served.socket, "B")
    try:
        async with app.run_test() as pilot:
            port = app.port
            assert isinstance(port, RemotePort)
            await until(pilot, lambda: port.session_starting)
            await pilot.press("escape")  # its own session's hook first
            await until(pilot, lambda: not port.session_starting)
            live = (await other.open_session(None, source="startup"))["session_id"]  # a second slow hook
            now = datetime.now().astimezone()
            assert await app._show_session(SessionInfo(live, Path(), "live", now, now, 0))  # attaches mid-hook
            assert port.session_starting
            await pilot.press(key)
            await until(pilot, lambda: not served.daemon.host(live).session_starting)
            await until(pilot, lambda: not port.session_starting)
            assert app.is_running
        assert all(_gone(int(pid)) for pid in pid_file.read_text().split())
    finally:
        await other.aclose()


@pytest.mark.parametrize("typed", ["hello", "/review x"])
async def test_a_message_or_command_typed_during_the_hook_gets_the_busy_line_and_stays_in_the_box(serve, tmp_path, typed: str) -> None:
    def setup(ws: Path) -> None:
        _slow_hook(tmp_path / "pids")(ws)
        _review(ws)

    served = await serve([REPLY], setup=setup)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        port = app.port
        assert isinstance(port, RemotePort)
        await until(pilot, lambda: port.session_starting)
        await type_and_send(pilot, typed)
        await until(pilot, lambda: any("busy — a session is loading" in t for t in notices(app)))
        await until(pilot, lambda: app.query_one(ChatInput).text == typed)
        await pilot.press("escape")
        await until(pilot, lambda: not port.session_starting)
    assert served.provider.requests == []


async def test_the_daemons_command_load_warnings_are_shown_once_at_launch(serve) -> None:
    def setup(ws: Path) -> None:
        _review(ws)
        (ws / ".agent" / "commands" / "model.md").write_text("clash")

    served = await serve([REPLY], setup=setup)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        port = await _open_port(app, pilot)
        await until(pilot, lambda: any("/model is a built-in command" in t for t in notices(app)))
        await app._load_daemon_commands(port)  # asked again (a reconnect): the same warning is not repeated
        assert sum("/model is a built-in command" in t for t in notices(app)) == 1


async def test_a_failed_command_listing_keeps_the_previous_set_and_warns_once(serve) -> None:
    from core.client.daemon import DaemonError

    served = await serve([REPLY], setup=_review)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        port = await _open_port(app, pilot)
        await until(pilot, lambda: "review" in app.commands.by_name)

        async def fail() -> object:
            raise DaemonError(500, "boom")

        port._http().commands = fail  # ty: ignore[invalid-assignment]
        await app._load_daemon_commands(port)
        assert "review" in app.commands.by_name
        assert sum("could not read the daemon's custom commands" in t for t in notices(app)) == 1


async def test_the_commands_are_read_again_after_a_reconnect_to_a_new_daemon(serve, monkeypatch) -> None:
    import src.tui.remote as remote

    monkeypatch.setattr(remote, "RECONNECT_WAIT", 5.0)
    monkeypatch.setattr(remote, "RETRY_STEP", 0.05)
    served = await serve([REPLY], setup=_review)
    app = AgentApp(RemotePort(served.socket, served.workspace, client_id="A"))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await until(pilot, lambda: "review" in app.commands.by_name)
        (served.workspace / ".agent" / "commands" / "second.md").write_text("Second")
        served.daemon.request_stop()
        await asyncio.wait_for(served.task, 10)
        await serve([REPLY])
        await until(pilot, lambda: "second" in app.commands.by_name, timeout=10)
