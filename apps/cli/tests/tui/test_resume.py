"""Starting with --continue / --resume: the saved session is loaded, redrawn and appended to."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from src.engine.session import FileRecorder, SessionError, SessionStore, SessionWriteError, sessions_root
from src.tui import StartChoice
from src.tui.styles import ICONS
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.tui.conftest import AppFactory, chat, settled, type_and_send, until

SaveSession = Callable[..., Path]
INFO = ICONS["info"]


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _files(tmp_path: Path) -> list[Path]:
    root = sessions_root(None)
    return sorted(root.rglob("*.jsonl")) if root.exists() else []


async def test_resume_by_a_unique_id_prefix(
    make_app: AppFactory, save_session: SaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter(["01JF3KAAAAAAAAAAAAAAAAAAAA", "01JF3MBBBBBBBBBBBBBBBBBBBB"])
    monkeypatch.setattr("src.engine.session.recorder.new_ulid", lambda: next(ids))
    save_session(chat("other", "other"), age=100)
    save_session(chat("wanted", "reply"), age=200)
    app = make_app(start=StartChoice("resume", "01jf3m"))  # any case, a prefix of one id only
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert [e.text for e in app.query(UserEntry)] == ["wanted"]


@pytest.mark.parametrize(
    ("prefix", "needle"),
    [("ZZZZZZZZ", "No saved session")],
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


@pytest.mark.parametrize("command", ["/new", "/compact", "/sessions"])
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
