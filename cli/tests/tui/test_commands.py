from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.mode import Mode
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import SessionStore, sessions_root
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import AssistantEntry, NoticeEntry, UserEntry

from tests.fakes import Gate, MemoryRecorder
from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


async def test_help_lists_the_keys_and_commands(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        (help_text,) = [n for n in _notices(app) if "Enter" in n]

    for needle in ("ctrl+j", "shift+enter", "Tab", "Esc", "ctrl+c", "/clear", "/new", "/sessions", "/compact", "/init", "/help", "/quit"):
        assert needle in help_text
    assert scripted(app).requests == []


async def test_help_lists_the_terminal_pane_keys(make_app: AppFactory) -> None:
    app = make_app(terminals=True)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/help")
        await pilot.pause()
        (help_text,) = [n for n in _notices(app) if "Enter" in n]

    for needle in ("ctrl+t", "terminal", "ctrl+\\", "ctrl+pageup", "ctrl+pagedown", "ctrl+w", "inside a terminal"):
        assert needle in help_text


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("doc", "needles"),
    [
        ("README.md", ("ctrl+t", "ctrl+\\", "ctrl+pageup", "ctrl+w", "Terminal.app", "term_start", "term_read", "term_stop")),
        (
            "AGENTS.md",
            ("ctrl+t", "ctrl+\\", "ctrl+pageup", "ctrl+w", "Terminal.app", "term_start", "term_read", "term_stop", "cli/src/agent/terminal/", "pyte"),
        ),
    ],
)
def test_the_docs_describe_the_terminal_pane_its_keys_and_tools(doc: str, needles: tuple[str, ...]) -> None:
    text = (ROOT / doc).read_text()

    missing = [needle for needle in needles if needle not in text]

    assert missing == []


async def test_an_unknown_command_gets_a_warning_and_no_request(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/nope")
        await pilot.pause()

        assert any(n.startswith("⚠") and "Unknown command" in n and "/nope" in n for n in _notices(app))
        assert scripted(app).requests == []


async def test_quit_command_exits(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/quit")
        await until(pilot, lambda: not app.is_running)


async def test_ctrl_c_when_idle_exits(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.is_running)


async def test_ctrl_c_during_a_turn_interrupts_instead_of_exiting(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("a"), gate, Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("ctrl+c")
        await settled(app)
        await pilot.pause()

        assert app.is_running
        assert any("interrupted" in n for n in _notices(app))
        assert not app.query_one(StatusBar).busy


@pytest.mark.parametrize("command", ["/clear", "/new"])
async def test_clear_and_new_empty_transcript_history_and_usage_and_start_a_new_session(
    make_app: AppFactory, command: str
) -> None:
    app = make_app([[TextDelta("hello"), Done(40, 10)]])
    async with app.run_test() as pilot:
        await pilot.press("tab")
        await type_and_send(pilot, "hi")
        await settled(app)
        await pilot.pause()
        assert app.query_one(StatusBar).used == 50
        assert app.agent.history
        assert app.agent.mode is Mode.PLAN

        await type_and_send(pilot, command)
        await pilot.pause()

        assert app.agent.history == []
        assert app.agent.mode is Mode.BUILD  # a new session starts like a fresh launch
        assert app.query_one(StatusBar).used == 0
        assert not app.query_one(StatusBar).estimated
        assert app.query_one(StatusBar).line.startswith("BUILD")
        assert list(app.query("#transcript > *")) == []


@pytest.mark.parametrize("command", ["/clear", "/new"])
async def test_the_next_message_after_clear_or_new_makes_a_second_file_and_leaves_the_first_alone(
    make_app: AppFactory, command: str
) -> None:
    app = make_app([[TextDelta("one"), Done(1, 1)], [TextDelta("two"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        (first,) = sorted(sessions_root(None).rglob("*.jsonl"))
        before = first.read_bytes()

        await type_and_send(pilot, command)
        await pilot.pause()
        assert sorted(sessions_root(None).rglob("*.jsonl")) == [first]  # lazy: no file for an empty session

        await type_and_send(pilot, "second")
        await settled(app)
        await pilot.pause()

        files = sorted(sessions_root(None).rglob("*.jsonl"))
        assert len(files) == 2 and first in files
        assert first.read_bytes() == before
        (second,) = [f for f in files if f != first]
        assert [m.content for m in SessionStore(sessions_root(None), app.workspace).load(second.stem).history] == [
            "second",
            "two",
        ]
        assert [(m.role, m.content) for m in scripted(app).requests[1][0][1:]] == [("user", "second")]


async def test_new_while_a_turn_runs_is_refused(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("a"), gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await type_and_send(pilot, "/new")
        await pilot.pause()

        assert any("busy" in n and "Esc" in n for n in _notices(app))
        assert app.agent.history  # untouched
        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert [e.text for e in app.query(UserEntry)] == ["go"]


async def test_clear_while_a_turn_runs_is_refused(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("a"), gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await type_and_send(pilot, "/clear")
        await pilot.pause()

        assert any("Esc" in n for n in _notices(app))
        assert app.agent.history  # untouched
        gate.release.set()
        await settled(app)
        await pilot.pause()
        # the notice sits between the two parts, so the reply is split in two entries
        assert "".join(e.text for e in app.query(AssistantEntry)) == "ab"


async def test_a_message_sent_while_busy_is_refused_with_a_notice(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await reached(gate)
        await type_and_send(pilot, "second")
        await pilot.pause()

        assert any("busy" in n and "Esc" in n for n in _notices(app))
        assert [e.text for e in app.query(UserEntry)] == ["first"]
        gate.release.set()
        await settled(app)
        assert len(scripted(app).requests) == 1
        assert isinstance(app.focused, ChatInput)


async def test_a_mode_toggle_mid_turn_shows_next_request_until_it_applies(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x\n")
    gate = Gate()
    read = ToolCallEvent(ToolCall("r1", "read", {"path": "a.py"}))
    app = make_app([[gate, read, Done(1, 1)], [TextDelta("second"), Done(1, 1)]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await reached(gate)
        await pilot.press("tab")
        assert status.line.startswith("PLAN (next request)")
        assert app.agent.mode is Mode.PLAN

        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert status.line.startswith("PLAN ·")
        first_tools, second_tools = (tools for _, tools in scripted(app).requests)
        assert "edit" in first_tools
        assert "edit" not in second_tools


async def test_a_message_refused_while_busy_stays_in_the_input(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await reached(gate)
        await type_and_send(pilot, "second")
        await pilot.pause()

        assert app.query_one(ChatInput).text == "second"
        assert [e.text for e in app.query(UserEntry)] == ["first"]
        gate.release.set()
        await settled(app)


async def test_leaving_the_app_closes_the_session_recorder(make_app: AppFactory) -> None:
    app = make_app()
    recorder = MemoryRecorder()
    async with app.run_test() as pilot:
        app.agent.reset(recorder=recorder)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.is_running)

    assert recorder.closed == 1


async def test_new_closes_the_old_recorder_off_the_ui_thread_before_the_next_message(make_app: AppFactory) -> None:
    import threading

    closed_on: list[threading.Thread] = []

    class ThreadRecorder(MemoryRecorder):
        def close(self) -> None:
            closed_on.append(threading.current_thread())
            super().close()

    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    old = ThreadRecorder()
    async with app.run_test() as pilot:
        app.agent.reset(recorder=old)  # the app's own session before /new
        await type_and_send(pilot, "/new")
        await until(pilot, lambda: old.closed == 1)
        await settled(app)

        assert closed_on and all(t is not threading.main_thread() for t in closed_on)
        await type_and_send(pilot, "hi")
        await settled(app)
        assert old.messages == []  # nothing reached the old session after /new
