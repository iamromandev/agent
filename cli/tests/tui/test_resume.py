"""Starting with --continue / --resume: the saved session is loaded, redrawn and appended to."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from src.agent.mode import Mode
from src.agent.provider.base import Done, Message, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import FileRecorder, SessionError, SessionStore, SessionWriteError, ToolMeta, sessions_root
from src.agent.session.paths import workspace_dir
from src.tui import StartChoice
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import ICONS, AssistantEntry, NoticeEntry, ToolEntry, UserEntry

from tests.tui.conftest import AppFactory, chat, scripted, settled, type_and_send, until

SaveSession = Callable[..., Path]
INFO = ICONS["info"]


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _files(tmp_path: Path) -> list[Path]:
    root = sessions_root(None)
    return sorted(root.rglob("*.jsonl")) if root.exists() else []


async def test_continue_shows_the_most_recently_updated_session(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(chat("older question", "older answer"), age=500)
    save_session(chat("newer question", "newer answer"), age=10)
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["newer question"]
        assert [e.text for e in app.query(AssistantEntry)] == ["newer answer"]
        status = app.query_one(StatusBar)
        assert status.line.startswith("BUILD")
        assert "ctx ~" in status.line and "/ 1000" in status.line
        assert status.estimated
        assert any(n.startswith(f'{INFO} Resumed "newer question" (2 messages)') for n in _notices(app))


async def test_a_follow_up_is_appended_to_the_same_file_and_the_model_sees_the_history(
    make_app: AppFactory, save_session: SaveSession, tmp_path: Path
) -> None:
    path = save_session(chat("first question", "first answer"))
    app = make_app([[TextDelta("second answer"), Done(30, 5)]], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "second question")
        await settled(app)
        await pilot.pause()

        (messages, _tools) = scripted(app).requests[0]
        assert [(m.role, m.content) for m in messages[1:]] == [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
        ]
        assert _files(tmp_path) == [path]
        loaded = SessionStore(sessions_root(None), tmp_path.resolve()).load(path.stem)
        assert [m.content for m in loaded.history] == [
            "first question",
            "first answer",
            "second question",
            "second answer",
        ]
        assert not app.query_one(StatusBar).estimated  # the model's own count replaced the guess
        assert app.query_one(StatusBar).used == 35


async def test_continue_without_a_session_shows_a_notice_and_starts_a_new_one(
    make_app: AppFactory, tmp_path: Path
) -> None:
    app = make_app([[TextDelta("hello"), Done(1, 1)]], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()
        assert f"{INFO} No previous session for this folder" in _notices(app)
        assert _files(tmp_path) == []

        await type_and_send(pilot, "hi")
        await settled(app)

    assert len(_files(tmp_path)) == 1


async def test_resume_by_a_unique_id_prefix(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter(["01JF3KAAAAAAAAAAAAAAAAAAAA", "01JF3MBBBBBBBBBBBBBBBBBBBB"])
    monkeypatch.setattr("src.agent.session.recorder.new_ulid", lambda: next(ids))
    save_session(chat("other", "other"), age=100)
    save_session(chat("wanted", "reply"), age=200)
    app = make_app(start=StartChoice("resume", "01jf3m"))  # any case, a prefix of one id only
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["wanted"]


@pytest.mark.parametrize(
    ("prefix", "needle"),
    [("ABC", "too short"), ("ZZZZZZZZ", "No saved session"), ("0", "too short")],
)
async def test_a_short_or_unknown_id_shows_an_error_naming_it_and_starts_a_new_session(
    make_app: AppFactory, save_session: SaveSession, prefix: str, needle: str
) -> None:
    save_session(chat("kept", "kept"))
    app = make_app(start=StartChoice("resume", prefix))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        (error,) = [n for n in _notices(app) if n.startswith("✗")]
        assert prefix in error and needle in error
        assert list(app.query(UserEntry)) == []
        assert app.agent.history == []


async def test_an_ambiguous_id_shows_an_error_and_starts_a_new_session(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter(["01JF3KAAAAAAAAAAAAAAAAAAAA", "01JF3KBBBBBBBBBBBBBBBBBBBB"])
    monkeypatch.setattr("src.agent.session.recorder.new_ulid", lambda: next(ids))
    save_session(chat("a", "a"), age=1)
    save_session(chat("b", "b"), age=2)
    app = make_app(start=StartChoice("resume", "01JF3K"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        (error,) = [n for n in _notices(app) if n.startswith("✗")]
        assert "01JF3K" in error and "matches 2 sessions" in error
        assert list(app.query(UserEntry)) == []


async def test_an_unreadable_session_file_is_skipped_with_a_warning(
    make_app: AppFactory, save_session: SaveSession, tmp_path: Path
) -> None:
    good = save_session(chat("good", "good"), age=50)
    (good.parent / "01HZZZZZZZZZZZZZZZZZZZZZZZ.jsonl").write_text("not a header\n")
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert "⚠ Skipped 1 unreadable session file(s)" in _notices(app)
        assert [e.text for e in app.query(UserEntry)] == ["good"]


async def test_a_resumed_edit_shows_collapsed_with_no_diff_in_build_mode_and_asks_again(
    make_app: AppFactory, save_session: SaveSession, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    call = ToolCall("c1", "edit", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
    save_session(
        [
            (Message("user", "change x"), None),
            (Message("assistant", "Editing.", [call]), None),
            (Message("tool", "replaced 1 occurrence(s) in a.py", tool_call_id="c1", name="edit"), ToolMeta("edit a.py", "replaced 1 occurrence(s)")),
            (Message("assistant", "Done, x is 2 now."), None),
        ]
    )
    app = make_app(
        [[ToolCallEvent(ToolCall("c2", "edit", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 3"})), Done(1, 1)]],
        start=StartChoice("continue"),
    )
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["change x"]
        assert [e.text for e in app.query(AssistantEntry)] == ["Editing.", "Done, x is 2 now."]
        (tool,) = app.query(ToolEntry)
        assert tool.title == "edit a.py · replaced 1 occurrence(s)"
        assert tool.collapsed and tool.detail == ""
        assert app.agent.mode is Mode.BUILD
        assert app.query_one(StatusBar).line.startswith("BUILD")

        await type_and_send(pilot, "again")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("escape")
        await settled(app)
        assert (tmp_path / "a.py").read_text() == "x = 1\n"  # nothing was applied without a yes


async def test_a_resumed_compaction_notice_is_redrawn(
    make_app: AppFactory, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    recorder = FileRecorder(workspace_dir(sessions_root(None), root), root)
    recorder.message(Message("user", "q"), None)
    recorder.message(Message("assistant", "a"), None)
    recorder.compaction([Message("user", "Summary of the earlier conversation:\nx")], {}, 0, 2, 900, 100)
    recorder.close()
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert any("Context compacted: pruned 0 tool outputs, summarized 2 messages (ctx ~900 → ~100)" in n for n in _notices(app))


async def test_the_session_is_loaded_in_a_worker_thread(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("q", "a"))
    threads: list[tuple[str, threading.Thread]] = []
    real_load, real_latest = SessionStore.load, SessionStore.latest

    def load(self: SessionStore, session_id: str):
        threads.append(("load", threading.current_thread()))
        return real_load(self, session_id)

    def latest(self: SessionStore):
        threads.append(("latest", threading.current_thread()))
        return real_latest(self)

    monkeypatch.setattr(SessionStore, "load", load)
    monkeypatch.setattr(SessionStore, "latest", latest)
    app = make_app(start=StartChoice("continue"))
    async with app.run_test():
        await settled(app)

    assert [name for name, _ in threads] == ["latest", "load"]
    assert all(thread is not threading.main_thread() for _, thread in threads)


async def test_a_session_that_cannot_be_loaded_shows_an_error_and_starts_a_new_one(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("q", "a"))

    def load(self: SessionStore, session_id: str):
        raise SessionError("Could not read session: boom")

    monkeypatch.setattr(SessionStore, "load", load)
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert "✗ Could not read session: boom" in _notices(app)
        assert list(app.query(UserEntry)) == []
        assert app.agent.history == []


async def test_a_recorder_that_cannot_be_resumed_leaves_a_new_session_and_an_error(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("q", "a"))

    def resume(cls: type[FileRecorder], path: Path, workspace: Path):
        raise SessionWriteError(2, "No such file or directory")

    monkeypatch.setattr(FileRecorder, "resume", classmethod(resume))
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert any(n.startswith("✗") and "No such file or directory" in n for n in _notices(app))
        assert list(app.query(UserEntry)) == []
        assert app.agent.history == []
        assert not app.query_one(StatusBar).estimated


async def test_a_message_sent_while_the_session_loads_is_refused_and_stays_in_the_input(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("q", "a"))
    entered, release = threading.Event(), threading.Event()
    real_load = SessionStore.load

    def slow_load(self: SessionStore, session_id: str):
        entered.set()
        assert release.wait(5.0), "the test never released the load"
        return real_load(self, session_id)

    monkeypatch.setattr(SessionStore, "load", slow_load)
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await until(pilot, entered.is_set)
        await type_and_send(pilot, "early")
        await pilot.pause()
        assert app.query_one(ChatInput).text == "early"
        assert scripted(app).requests == []
        assert "⚠ busy — a session is loading" in _notices(app)

        release.set()
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["q"]
        assert app.query_one(ChatInput).text == "early"  # still there to send again


async def test_opening_and_closing_without_a_message_writes_nothing(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.press("ctrl+c")
        await until(pilot, lambda: not app.is_running)

    assert not sessions_root(None).exists() or list(sessions_root(None).rglob("*")) == []


async def test_a_session_adds_nothing_to_the_workspace(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    app = make_app([[TextDelta("hello"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hi")
        await settled(app)

    assert len(_files(tmp_path)) == 1  # a session was saved, elsewhere
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


# --- the start worker holds the session switch from the first line ------------------------------


def _gate_store(monkeypatch: pytest.MonkeyPatch, name: str) -> tuple[threading.Event, threading.Event]:
    """Block `SessionStore.<name>` (it runs in a thread) until released; returns (entered, release)."""
    entered, release = threading.Event(), threading.Event()
    real = getattr(SessionStore, name)

    def gated(self: SessionStore, *args: object):
        entered.set()
        assert release.wait(5.0), "the test never released the store"
        return real(self, *args)

    monkeypatch.setattr(SessionStore, name, gated)
    return entered, release


async def test_a_message_sent_while_the_latest_session_is_looked_up_is_refused_and_the_resume_goes_on(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_session(chat("q", "a"))
    entered, release = _gate_store(monkeypatch, "latest")
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await until(pilot, entered.is_set)
        await type_and_send(pilot, "early")
        await pilot.pause()
        assert app.query_one(ChatInput).text == "early"
        assert scripted(app).requests == []
        assert "⚠ busy — a session is loading" in _notices(app)

        release.set()
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["q"]
        assert not any("busy — press Esc" in n for n in _notices(app))
        assert [m.content for m in app.agent.history] == ["q", "a"]


async def test_a_message_sent_while_a_session_is_found_by_id_is_refused(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save_session(chat("q", "a"))
    entered, release = _gate_store(monkeypatch, "find")
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("resume", path.stem[:8]))
    async with app.run_test() as pilot:
        await until(pilot, entered.is_set)
        await type_and_send(pilot, "early")
        await pilot.pause()
        assert scripted(app).requests == []

        release.set()
        await settled(app)
        await pilot.pause()

        assert [m.content for m in app.agent.history] == ["q", "a"]


@pytest.mark.parametrize("command", ["/new", "/clear", "/compact", "/sessions"])
async def test_commands_are_refused_while_the_start_session_is_looked_up(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    save_session(chat("q", "a"))
    entered, release = _gate_store(monkeypatch, "latest")
    app = make_app(start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await until(pilot, entered.is_set)
        await type_and_send(pilot, command)
        await pilot.pause()
        assert any(n.startswith("⚠ busy") for n in _notices(app))

        release.set()
        await settled(app)
        await pilot.pause()

        assert [m.content for m in app.agent.history] == ["q", "a"]  # the resume was not replaced


async def test_after_a_resume_that_finds_nothing_a_message_is_accepted(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("continue"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()
        assert f"{INFO} No previous session for this folder" in _notices(app)

        await type_and_send(pilot, "hello")
        await settled(app)

        assert [e.text for e in app.query(UserEntry)] == ["hello"]


@pytest.mark.parametrize("known", [False, True])
async def test_after_a_resume_that_fails_a_message_is_accepted(
    make_app: AppFactory, save_session: SaveSession, known: bool
) -> None:
    if known:  # two files sharing a prefix: the id is ambiguous
        save_session(chat("q", "a"))
        save_session(chat("r", "b"))
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("resume", "0" if known else "nope"))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()
        assert any(n.startswith("✗") for n in _notices(app))

        await type_and_send(pilot, "hello")
        await settled(app)

        assert [e.text for e in app.query(UserEntry)] == ["hello"]


async def test_bare_resume_refuses_input_until_the_picker_resolves_and_escape_releases_it(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tui.screens.sessions import SessionPicker

    save_session(chat("q", "a"))
    entered, release = _gate_store(monkeypatch, "list")
    app = make_app([[TextDelta("ok"), Done(1, 1)]], start=StartChoice("pick"))
    async with app.run_test() as pilot:
        await until(pilot, entered.is_set)
        await type_and_send(pilot, "early")
        await pilot.pause()
        assert scripted(app).requests == []
        assert "⚠ busy — a session is loading" in _notices(app)

        release.set()
        await until(pilot, lambda: isinstance(app.screen, SessionPicker))
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()
        assert not isinstance(app.screen, SessionPicker)

        app.query_one(ChatInput).clear()
        await type_and_send(pilot, "later")
        await settled(app)

        assert [e.text for e in app.query(UserEntry)] == ["later"]
