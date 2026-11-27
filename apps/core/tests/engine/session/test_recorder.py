from __future__ import annotations

import errno
import json
import os
import stat
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from core.engine.provider.base import Message
from core.engine.session.paths import workspace_dir
from core.engine.session.recorder import FileRecorder, NullRecorder, SessionWriteError
from core.engine.session.records import ToolMeta
from core.engine.usage import UsageRecord

NOW = datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC)
AC3_TEXT = "  Add   a --verbose\nflag to health check and also update the docs please  "


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def directory(tmp_path: Path, workspace: Path) -> Path:
    return workspace_dir(tmp_path / "root", workspace)


@pytest.fixture
def recorder(directory: Path, workspace: Path) -> Iterator[FileRecorder]:
    rec = FileRecorder(directory, workspace, now=lambda: NOW)
    yield rec
    rec.close()


def lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    assert text == "" or text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def raw_lines(path: Path) -> list[str]:
    return path.read_bytes().decode().splitlines()


@contextmanager
def umask(value: int) -> Generator[None]:
    old = os.umask(value)
    try:
        yield
    finally:
        os.umask(old)


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_a_chosen_id_is_known_before_the_first_record_and_names_the_file(directory: Path, workspace: Path) -> None:
    chosen = "01KCHOSENCHOSENCHOSENCHOSE"
    rec = FileRecorder(directory, workspace, session_id=chosen)
    assert rec.session_id == chosen and rec.path is None  # no file yet
    rec.message(Message("user", "hi"), None)
    rec.close()
    assert rec.path == directory / f"{chosen}.jsonl"
    assert json.loads(rec.path.read_text().splitlines()[0])["id"] == chosen


def test_a_first_message_that_is_not_from_the_user_writes_no_title(recorder: FileRecorder) -> None:
    recorder.message(Message("assistant", "hello"), None)
    assert recorder.path is not None
    assert [r["type"] for r in lines(recorder.path)] == ["session", "message"]


def failing_write(
    monkeypatch: pytest.MonkeyPatch, fail_on: set[int], *, short: bool = False, zero: bool = False
) -> list[int]:
    """Make the n-th os.write call after this patch (1-based) fail. Returns the attempted call numbers."""
    real_write = os.write
    calls: list[int] = []

    def fake(fd: int, data: Any) -> int:
        calls.append(len(calls) + 1)
        if calls[-1] in fail_on:
            if zero:
                return 0
            if short:
                return real_write(fd, bytes(data)[: len(data) // 2])
            raise OSError(28, "No space left on device")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", fake)
    return calls


def test_a_short_write_is_an_error_and_the_line_is_completed_without_duplicates(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    recorder.message(Message("user", "one"), None)
    failing_write(monkeypatch, {1}, short=True)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("assistant", "two\nlines of 日本語"), None)
    recorder.message(Message("user", "three"), None)

    assert recorder.path is not None
    records = lines(recorder.path)  # every line parses: no torn or doubled record
    assert [r["content"] for r in records if r["type"] == "message"] == [
        "one",
        "two\nlines of 日本語",
        "three",
    ]


def test_a_zero_byte_write_is_an_error(monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder) -> None:
    failing_write(monkeypatch, {1}, zero=True)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "one"), None)
    recorder.message(Message("user", "two"), None)
    assert recorder.path is not None
    assert [r["content"] for r in lines(recorder.path) if r["type"] == "message"] == ["one", "two"]


def test_a_model_switch_is_recorded_with_its_window(recorder: FileRecorder) -> None:
    recorder.message(Message("user", "q"), None)
    recorder.model("qwen3:14b", 8192)

    assert recorder.path is not None
    record = lines(recorder.path)[-1]
    assert list(record) == ["type", "at", "model", "window"]
    assert (record["type"], record["model"], record["window"]) == ("model", "qwen3:14b", 8192)


def test_close_swallows_a_write_error(monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder) -> None:
    failing_write(monkeypatch, {1, 2})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)
    recorder.close()


def test_null_recorder_does_no_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("NullRecorder touched the filesystem")

    for name in ("open", "write", "mkdir"):
        monkeypatch.setattr(os, name, forbidden)
    null = NullRecorder()
    null.message(Message("user", "a"), None)
    null.compaction([], {}, 0, 0, 0, 0)
    null.undo(0)
    null.model("a", 1)
    null.close()
    assert list(tmp_path.iterdir()) == []


def test_a_failure_building_the_message_record_leaves_nothing_started(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    import core.engine.session.recorder as recorder_module

    def broken(m: Message, meta: ToolMeta | None) -> dict[str, Any]:
        raise RuntimeError("cannot build the message")

    monkeypatch.setattr(recorder_module, "message_to_obj", broken)
    with pytest.raises(RuntimeError):
        recorder.message(Message("user", "one"), None)
    assert recorder.path is None
    assert not recorder._pending


def test_resume_of_a_missing_file_raises_and_creates_nothing(tmp_path: Path, workspace: Path) -> None:
    directory = tmp_path / "root" / "abc"
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    resumed = FileRecorder.resume(path, workspace)
    with pytest.raises(SessionWriteError) as excinfo:
        resumed.message(Message("user", "one"), None)
    assert excinfo.value.errno == errno.ENOENT
    assert not path.exists()
    assert not directory.exists()


# --- symlinks, ownership and folder modes -------------------------------------------------------


def snapshot(path: Path) -> tuple[int, list[str]]:
    return mode(path), sorted(p.name for p in path.iterdir())


def test_a_workspace_folder_replaced_by_a_symlink_is_refused_and_the_target_is_untouched(
    tmp_path: Path, directory: Path, workspace: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    target.chmod(0o755)
    (target / "keep.txt").write_text("x")
    directory.parent.mkdir(parents=True)
    directory.symlink_to(target)
    before = snapshot(target)

    rec = FileRecorder(directory, workspace)
    with pytest.raises(SessionWriteError):
        rec.message(Message("user", "secret"), None)
    assert snapshot(target) == before
    assert (target / "keep.txt").read_text() == "x"


def test_a_sessions_root_that_is_a_symlink_is_refused(tmp_path: Path, directory: Path, workspace: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    target.chmod(0o755)
    directory.parent.symlink_to(target)
    rec = FileRecorder(directory, workspace)
    with pytest.raises(SessionWriteError):
        rec.message(Message("user", "secret"), None)
    assert snapshot(target) == (0o755, [])


def test_resuming_a_symlinked_session_file_is_refused_and_the_target_is_untouched(
    tmp_path: Path, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    target = tmp_path / "target.jsonl"
    target.write_bytes(b'{"type":"session"}\n')
    target.chmod(0o644)
    link = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    link.symlink_to(target)

    resumed = FileRecorder.resume(link, workspace)
    with pytest.raises(SessionWriteError):
        resumed.message(Message("user", "secret"), None)
    assert target.read_bytes() == b'{"type":"session"}\n'
    assert mode(target) == 0o644


def test_resuming_a_fifo_is_refused(directory: Path, workspace: Path) -> None:
    directory.mkdir(parents=True)
    fifo = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    os.mkfifo(fifo)
    resumed = FileRecorder.resume(fifo, workspace)
    with pytest.raises(SessionWriteError):
        resumed.message(Message("user", "x"), None)


def test_a_session_file_owned_by_someone_else_is_refused(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    path.write_bytes(b'{"type":"session"}\n')
    monkeypatch.setattr(os, "getuid", lambda: os.stat(path).st_uid + 1)
    resumed = FileRecorder.resume(path, workspace)
    with pytest.raises(SessionWriteError):
        resumed.message(Message("user", "x"), None)
    assert path.read_bytes() == b'{"type":"session"}\n'


def test_a_failed_newline_repair_is_retried_before_any_record(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    path.write_bytes(b'{"type":"session"}\n{"type":"message","role":"us')
    failing_write(monkeypatch, {1})  # the repair newline
    resumed = FileRecorder.resume(path, workspace)
    with pytest.raises(SessionWriteError):
        resumed.message(Message("user", "new"), None)
    assert path.read_bytes() == b'{"type":"session"}\n{"type":"message","role":"us'
    resumed.message(Message("user", "again"), None)
    resumed.close()
    assert path.read_bytes().count(b"\n") == 4  # header, repaired line, "new", "again"
    assert [json.loads(line)["content"] for line in path.read_bytes().decode().splitlines()[2:]] == ["new", "again"]


def _usage(model: str = "m", i: int = 100) -> UsageRecord:
    return UsageRecord(model, i, 20, 1500, False, "2026-11-18T10:00:00Z")


def test_usage_before_any_message_writes_no_file(recorder: FileRecorder, directory: Path) -> None:
    recorder.usage(_usage())

    assert recorder.path is None
    assert not directory.exists() or list(directory.iterdir()) == []
