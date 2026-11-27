"""The session lock: one process at a time may have a session's recorder open."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from core.engine.provider.base import Message
from core.engine.session import FileRecorder, SessionError, SessionStore, sessions_root, workspace_dir
from core.engine.session.lock import SessionLock, SessionLocked
from core.testing.helpers import session_held_elsewhere


def _saved_session(root: Path) -> tuple[Path, str]:
    """A workspace with one saved session; returns the sessions folder and the id."""
    ws = root / "ws"
    ws.mkdir()
    folder = workspace_dir(sessions_root(None), ws.resolve())
    recorder = FileRecorder(folder, ws)
    recorder.message(Message("user", "hello"), None)
    session_id = recorder.session_id
    recorder.close()
    return folder, session_id


def test_a_session_open_in_another_process_is_refused_and_marked(tmp_path: Path) -> None:
    folder, session_id = _saved_session(tmp_path)
    with session_held_elsewhere(folder, session_id):
        with pytest.raises(SessionLocked, match=f"session {session_id} is open in another agent process"):
            FileRecorder.resume(folder / f"{session_id}.jsonl", tmp_path / "ws")
        (info,) = SessionStore(sessions_root(None), (tmp_path / "ws").resolve()).list()
    assert issubclass(SessionLocked, SessionError)  # every opener already shows a SessionError and stays put
    assert info.open_elsewhere


def test_the_lock_goes_with_close_and_this_processs_own_session_is_not_open_elsewhere(tmp_path: Path) -> None:
    folder, session_id = _saved_session(tmp_path)
    recorder = FileRecorder.resume(folder / f"{session_id}.jsonl", tmp_path / "ws")
    lock_file = folder / f"{session_id}.lock"
    assert stat.S_IMODE(os.lstat(lock_file).st_mode) == 0o600
    (info,) = SessionStore(sessions_root(None), (tmp_path / "ws").resolve()).list()
    assert not info.open_elsewhere
    recorder.close()

    again = FileRecorder.resume(folder / f"{session_id}.jsonl", tmp_path / "ws")  # free again
    again.close()


def test_a_new_session_is_locked_once_its_file_exists(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    folder = workspace_dir(sessions_root(None), ws.resolve())
    recorder = FileRecorder(folder, ws)
    recorder.message(Message("user", "hello"), None)
    assert SessionLock.held_by_us(folder, recorder.session_id)
    recorder.close()
    assert not SessionLock.held_by_us(folder, recorder.session_id)


def test_a_symlinked_lock_file_is_refused(tmp_path: Path) -> None:
    folder, session_id = _saved_session(tmp_path)
    (folder / f"{session_id}.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError):
        SessionLock.acquire(folder, session_id)
