from __future__ import annotations

import errno
import json
import os
import re
import stat
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from src.agent.provider.base import Message, ToolCall
from src.agent.session.paths import workspace_dir
from src.agent.session.recorder import FileRecorder, NullRecorder, SessionWriteError, title_from
from src.agent.session.records import ToolMeta, message_from_obj

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


def test_nothing_exists_before_the_first_message(recorder: FileRecorder, directory: Path) -> None:
    assert recorder.path is None
    assert not directory.exists()
    assert not directory.parent.exists()


def test_the_first_message_creates_one_ulid_file_under_the_directory(
    recorder: FileRecorder, directory: Path
) -> None:
    recorder.message(Message("user", "hi"), None)
    assert recorder.path is not None
    assert recorder.path.parent == directory
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}\.jsonl", recorder.path.name)
    assert list(directory.iterdir()) == [recorder.path]


def test_the_file_holds_the_header_then_the_title_then_the_message(
    recorder: FileRecorder, workspace: Path
) -> None:
    recorder.message(Message("user", AC3_TEXT), None)
    assert recorder.path is not None
    header, title, message = lines(recorder.path)
    assert header == {
        "type": "session",
        "version": 1,
        "id": recorder.path.stem,
        "workspace": str(workspace.resolve()),
        "created_at": "2026-11-12T15:03:18Z",
    }
    assert title == {"type": "title", "title": title_from(AC3_TEXT), "at": "2026-11-12T15:03:18Z"}
    assert title["title"] == "Add a --verbose flag to health check and also upd…"
    assert message["type"] == "message" and message["role"] == "user"
    assert message["content"] == AC3_TEXT
    assert message["at"] == "2026-11-12T15:03:18Z"


def test_later_messages_are_appended_in_order_without_a_second_header(recorder: FileRecorder) -> None:
    recorder.message(Message("user", "q"), None)
    call = ToolCall("c1", "read", {"path": "a.py"})
    recorder.message(Message("assistant", "", tool_calls=[call]), None)
    recorder.message(
        Message("tool", "x = 1", tool_call_id="c1", name="read"), ToolMeta("read a.py", "1 lines")
    )
    recorder.message(Message("assistant", "done"), None)
    assert recorder.path is not None
    records = lines(recorder.path)
    assert [r["type"] for r in records] == ["session", "title", "message", "message", "message", "message"]
    assert [r["role"] for r in records[2:]] == ["user", "assistant", "tool", "assistant"]
    tool_message, meta = message_from_obj(records[4])
    assert tool_message == Message("tool", "x = 1", tool_call_id="c1", name="read")
    assert meta == ToolMeta("read a.py", "1 lines")
    assert message_from_obj(records[3])[0].tool_calls == [call]


def test_a_first_message_that_is_not_from_the_user_writes_no_title(recorder: FileRecorder) -> None:
    recorder.message(Message("assistant", "hello"), None)
    assert recorder.path is not None
    assert [r["type"] for r in lines(recorder.path)] == ["session", "message"]


def test_resume_appends_to_the_existing_file(
    recorder: FileRecorder, directory: Path, workspace: Path
) -> None:
    recorder.message(Message("user", "first"), None)
    path = recorder.path
    assert path is not None
    recorder.close()

    resumed = FileRecorder.resume(path, workspace)
    assert resumed.path == path
    resumed.message(Message("user", "second"), None)
    resumed.close()

    records = lines(path)
    assert [r["type"] for r in records] == ["session", "title", "message", "message"]
    assert [r["content"] for r in records[2:]] == ["first", "second"]
    assert list(directory.iterdir()) == [path]


def test_resume_is_lazy_and_never_writes_a_header(tmp_path: Path, workspace: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"type":"session"}\n')
    resumed = FileRecorder.resume(path, workspace)
    assert path.read_text() == '{"type":"session"}\n'
    resumed.message(Message("user", "u"), None)
    resumed.close()
    assert [r["type"] for r in lines(path)] == ["session", "message"]


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o077])
def test_modes_do_not_depend_on_the_umask(directory: Path, workspace: Path, mask: int) -> None:
    with umask(mask):
        rec = FileRecorder(directory, workspace)
        rec.message(Message("user", "secret"), None)
        rec.close()
    assert rec.path is not None
    assert mode(directory) == 0o700
    assert mode(directory.parent) == 0o700
    assert mode(rec.path) == 0o600


def test_a_resumed_file_keeps_mode_0600(directory: Path, workspace: Path) -> None:
    with umask(0):
        rec = FileRecorder(directory, workspace)
        rec.message(Message("user", "a"), None)
        rec.close()
        assert rec.path is not None
        rec.path.chmod(0o644)  # someone loosened it
        resumed = FileRecorder.resume(rec.path, workspace)
        resumed.message(Message("user", "b"), None)
        resumed.close()
    assert mode(rec.path) == 0o600


def test_an_existing_loose_directory_is_tightened(directory: Path, workspace: Path) -> None:
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    rec = FileRecorder(directory, workspace)
    rec.message(Message("user", "a"), None)
    rec.close()
    assert mode(directory) == 0o700


def test_the_workspace_is_never_written(directory: Path, workspace: Path) -> None:
    (workspace / "keep.txt").write_text("x")
    before = sorted(p.name for p in workspace.rglob("*"))
    rec = FileRecorder(directory, workspace)
    rec.message(Message("user", "hi"), None)
    rec.close()
    assert sorted(p.name for p in workspace.rglob("*")) == before


def test_each_record_is_one_os_write_of_the_whole_line_on_an_append_fd(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    real_write, real_open = os.write, os.open
    writes: list[bytes] = []
    flags: list[int] = []

    def spy_write(fd: int, data: Any) -> int:
        writes.append(bytes(data))
        return real_write(fd, data)

    def spy_open(path: Any, flag: int, *args: Any, **kwargs: Any) -> int:
        flags.append(flag)
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(os, "write", spy_write)
    monkeypatch.setattr(os, "open", spy_open)
    recorder.message(Message("user", "hi\nthere"), None)
    recorder.message(Message("assistant", "ok"), None)

    assert len(writes) == 4  # header, title, two messages
    assert all(w.endswith(b"\n") and w.count(b"\n") == 1 for w in writes)
    assert flags and flags[0] & os.O_APPEND and flags[0] & os.O_CREAT and flags[0] & (os.O_WRONLY | os.O_RDWR)
    assert recorder.path is not None
    assert recorder.path.read_bytes() == b"".join(writes)


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


def test_a_failed_write_raises_and_the_next_call_retries_in_order_without_duplicates(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    recorder.message(Message("user", "one"), None)
    failing_write(monkeypatch, {1})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("assistant", "two"), None)
    recorder.message(Message("user", "three"), None)

    assert recorder.path is not None
    contents = [r["content"] for r in lines(recorder.path) if r["type"] == "message"]
    assert contents == ["one", "two", "three"]


def test_an_error_on_the_first_message_keeps_header_title_and_message_queued(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    failing_write(monkeypatch, {1})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "one"), None)
    recorder.message(Message("assistant", "two"), None)

    assert recorder.path is not None
    records = lines(recorder.path)
    assert [r["type"] for r in records] == ["session", "title", "message", "message"]
    assert [r["content"] for r in records[2:]] == ["one", "two"]


def test_several_failures_in_a_row_keep_the_queue_in_order(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    failing_write(monkeypatch, {1, 2, 3})
    for text in ("a", "b", "c"):
        with pytest.raises(SessionWriteError):
            recorder.message(Message("user", text), None)
    recorder.message(Message("user", "d"), None)

    assert recorder.path is not None
    assert [r["content"] for r in lines(recorder.path) if r["type"] == "message"] == ["a", "b", "c", "d"]
    assert sum(r["type"] == "session" for r in lines(recorder.path)) == 1


def test_a_failure_in_the_middle_of_the_queue_keeps_the_rest_queued(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    failing_write(monkeypatch, {1, 2})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)  # call 1 fails: all three queued
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "b"), None)  # call 2 fails: header still first
    recorder.message(Message("user", "c"), None)

    assert recorder.path is not None
    assert [r["type"] for r in lines(recorder.path)] == ["session", "title", "message", "message", "message"]


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


def test_a_failure_to_open_the_file_raises_and_is_retried(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    real_open = os.open
    attempts = {"n": 0}

    def flaky_open(path: Any, flag: int, *args: Any, **kwargs: Any) -> int:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise PermissionError(13, "denied")
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(os, "open", flaky_open)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "one"), None)
    recorder.message(Message("user", "two"), None)

    assert recorder.path is not None
    records = lines(recorder.path)
    assert [r["type"] for r in records] == ["session", "title", "message", "message"]
    assert [r["content"] for r in records[2:]] == ["one", "two"]


def test_a_failure_to_create_the_directory_raises_a_write_error(tmp_path: Path, workspace: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    rec = FileRecorder(blocker / "ws", workspace)
    with pytest.raises(SessionWriteError):
        rec.message(Message("user", "one"), None)


def test_the_write_error_is_an_oserror() -> None:
    assert issubclass(SessionWriteError, OSError)


def test_compaction_records_the_compacted_history_with_counts(recorder: FileRecorder) -> None:
    recorder.message(Message("user", "q"), None)
    history = [
        Message("user", "Summary of the earlier conversation:\nstuff"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "read", {"path": "a"})]),
        Message("tool", "[output pruned: read a · 1 lines]", tool_call_id="c1", name="read"),
    ]
    recorder.compaction(history, {"c1": ToolMeta("read a", "1 lines")}, 3, 5, 900, 400)

    assert recorder.path is not None
    record = lines(recorder.path)[-1]
    assert list(record) == ["type", "history", "pruned", "summarized", "before", "after", "at"]
    assert record["type"] == "compaction"
    assert (record["pruned"], record["summarized"], record["before"], record["after"]) == (3, 5, 900, 400)
    restored = [message_from_obj(obj, embedded=True) for obj in record["history"]]
    assert [m for m, _ in restored] == history
    assert [meta for _, meta in restored] == [None, None, ToolMeta("read a", "1 lines")]
    assert all("type" not in obj and "at" not in obj for obj in record["history"])


def test_a_failed_compaction_write_stays_queued_before_later_messages(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    recorder.message(Message("user", "q"), None)
    failing_write(monkeypatch, {1})
    with pytest.raises(SessionWriteError):
        recorder.compaction([Message("user", "kept")], {}, 1, 0, 10, 5)
    recorder.message(Message("user", "later"), None)

    assert recorder.path is not None
    assert [r["type"] for r in lines(recorder.path)] == ["session", "title", "message", "compaction", "message"]


def test_close_is_idempotent_and_safe_before_any_write(recorder: FileRecorder) -> None:
    recorder.close()
    recorder.close()
    recorder.message(Message("user", "a"), None)
    recorder.close()
    recorder.close()


def test_close_releases_the_file_descriptor(monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder) -> None:
    real_close = os.close
    closed: list[int] = []
    monkeypatch.setattr(os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    recorder.message(Message("user", "a"), None)
    recorder.close()
    recorder.close()
    assert len(closed) == 1


def test_a_message_after_close_still_appends_to_the_same_file(recorder: FileRecorder) -> None:
    recorder.message(Message("user", "a"), None)
    recorder.close()
    recorder.message(Message("user", "b"), None)
    assert recorder.path is not None
    assert [r["content"] for r in lines(recorder.path) if r["type"] == "message"] == ["a", "b"]
    assert sum(r["type"] == "session" for r in lines(recorder.path)) == 1


def test_close_makes_one_last_attempt_to_write_queued_records(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    failing_write(monkeypatch, {1})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)
    recorder.close()  # must not raise
    assert recorder.path is not None
    assert [r["content"] for r in lines(recorder.path) if r["type"] == "message"] == ["a"]


def test_close_swallows_a_write_error(monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder) -> None:
    failing_write(monkeypatch, {1, 2})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)
    recorder.close()


def test_the_default_clock_gives_a_utc_timestamp(directory: Path, workspace: Path) -> None:
    rec = FileRecorder(directory, workspace)
    rec.message(Message("user", "a"), None)
    rec.close()
    assert rec.path is not None
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lines(rec.path)[0]["created_at"])


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
    null.close()
    assert list(tmp_path.iterdir()) == []


def test_recorder_factory_type_is_usable() -> None:
    factory: Callable[[], NullRecorder] = NullRecorder
    assert isinstance(factory(), NullRecorder)


def test_a_failure_building_the_first_records_leaves_nothing_started(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    import src.agent.session.recorder as recorder_module

    real_dumps = recorder_module.dumps
    calls = {"n": 0}

    def flaky_dumps(obj: dict[str, Any]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:  # the header
            raise RuntimeError("cannot build the header")
        return real_dumps(obj)

    monkeypatch.setattr(recorder_module, "dumps", flaky_dumps)
    with pytest.raises(RuntimeError):
        recorder.message(Message("user", "one"), None)
    assert recorder.path is None

    recorder.message(Message("user", "two"), None)
    assert recorder.path is not None
    records = lines(recorder.path)
    assert [r["type"] for r in records] == ["session", "title", "message"]
    assert records[0]["id"] == recorder.path.stem
    assert [r["content"] for r in records[2:]] == ["two"]


def test_a_failure_building_the_message_record_leaves_nothing_started(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    import src.agent.session.recorder as recorder_module

    def broken(m: Message, meta: ToolMeta | None) -> dict[str, Any]:
        raise RuntimeError("cannot build the message")

    monkeypatch.setattr(recorder_module, "message_to_obj", broken)
    with pytest.raises(RuntimeError):
        recorder.message(Message("user", "one"), None)
    assert recorder.path is None
    assert not recorder._pending


def test_resume_of_a_file_that_ends_without_a_newline_writes_one_first(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    path.write_bytes(b'{"type":"session"}\n{"type":"message","role":"us')
    real_write = os.write
    writes: list[bytes] = []

    def spy(fd: int, data: Any) -> int:
        writes.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", spy)
    resumed = FileRecorder.resume(path, workspace)
    resumed.message(Message("user", "new"), None)
    resumed.close()

    assert writes[0] == b"\n"
    assert len(writes) == 2
    assert path.read_bytes().startswith(b'{"type":"session"}\n{"type":"message","role":"us\n{"type":"message"')


def test_resume_of_a_file_that_ends_with_a_newline_writes_no_extra_one(directory: Path, workspace: Path) -> None:
    directory.mkdir(parents=True)
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    path.write_bytes(b'{"type":"session"}\n')
    resumed = FileRecorder.resume(path, workspace)
    resumed.message(Message("user", "new"), None)
    resumed.close()
    assert b"\n\n" not in path.read_bytes()


def test_a_reopen_after_close_repairs_a_tail_cut_in_the_meantime(directory: Path, workspace: Path) -> None:
    rec = FileRecorder(directory, workspace, now=lambda: NOW)
    rec.message(Message("user", "one"), None)
    rec.close()
    assert rec.path is not None
    rec.path.write_bytes(rec.path.read_bytes()[:-5])
    rec.message(Message("user", "two"), None)
    rec.close()
    text = rec.path.read_bytes().decode()
    assert text.endswith("\n")
    assert json.loads(text.splitlines()[-1])["content"] == "two"


def test_resume_of_a_missing_file_raises_and_creates_nothing(tmp_path: Path, workspace: Path) -> None:
    directory = tmp_path / "root" / "abc"
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    resumed = FileRecorder.resume(path, workspace)
    with pytest.raises(SessionWriteError) as excinfo:
        resumed.message(Message("user", "one"), None)
    assert excinfo.value.errno == errno.ENOENT
    assert not path.exists()
    assert not directory.exists()


def test_resume_of_a_missing_file_in_an_existing_folder_creates_no_file(directory: Path, workspace: Path) -> None:
    directory.mkdir(parents=True)
    path = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    resumed = FileRecorder.resume(path, workspace)
    with pytest.raises(SessionWriteError) as excinfo:
        resumed.message(Message("user", "one"), None)
    assert excinfo.value.errno == errno.ENOENT
    assert list(directory.iterdir()) == []


def test_a_file_deleted_after_close_is_not_recreated_headerless(directory: Path, workspace: Path) -> None:
    rec = FileRecorder(directory, workspace)
    rec.message(Message("user", "one"), None)
    rec.close()
    assert rec.path is not None
    rec.path.unlink()
    with pytest.raises(SessionWriteError) as excinfo:
        rec.message(Message("user", "two"), None)
    assert excinfo.value.errno == errno.ENOENT
    assert not rec.path.exists()


def test_a_new_session_whose_file_name_already_exists_is_an_error_and_keeps_the_file(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    taken = directory / "01AAAAAAAAAAAAAAAAAAAAAAAA.jsonl"
    taken.write_bytes(b'{"type":"session"}\n')
    monkeypatch.setattr("src.agent.session.recorder.new_ulid", lambda: "01AAAAAAAAAAAAAAAAAAAAAAAA")
    rec = FileRecorder(directory, workspace)
    with pytest.raises(SessionWriteError) as excinfo:
        rec.message(Message("user", "one"), None)
    assert excinfo.value.errno == errno.EEXIST
    assert taken.read_bytes() == b'{"type":"session"}\n'


def test_a_new_session_can_reopen_its_own_file_when_the_header_was_never_written(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    failing_write(monkeypatch, {1})
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "one"), None)
    recorder.close()  # the fd is released with the header still pending
    recorder.message(Message("user", "two"), None)
    assert recorder.path is not None
    assert [r["type"] for r in lines(recorder.path)] == ["session", "title", "message", "message"]


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


def test_a_workspace_folder_owned_by_someone_else_is_refused(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    directory.mkdir(parents=True)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(directory).st_uid + 1)
    rec = FileRecorder(directory, workspace)
    with pytest.raises(SessionWriteError):
        rec.message(Message("user", "x"), None)
    assert list(directory.iterdir()) == []


def test_the_session_file_is_opened_without_following_symlinks(
    monkeypatch: pytest.MonkeyPatch, recorder: FileRecorder
) -> None:
    real_open = os.open
    flags: list[int] = []

    def spy(path: Any, flag: int, *args: Any, **kwargs: Any) -> int:
        flags.append(flag)
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    recorder.message(Message("user", "hi"), None)
    assert flags[0] & os.O_NOFOLLOW
    assert flags[0] & os.O_CLOEXEC


def test_missing_ancestors_are_created_0700_whatever_the_umask(tmp_path: Path, workspace: Path) -> None:
    root = tmp_path / "a" / "b" / "root"
    with umask(0):
        rec = FileRecorder(workspace_dir(root, workspace), workspace)
        rec.message(Message("user", "x"), None)
        rec.close()
    for created in (tmp_path / "a", tmp_path / "a" / "b", root, workspace_dir(root, workspace)):
        assert mode(created) == 0o700


def test_existing_ancestors_are_left_alone(tmp_path: Path, workspace: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    loose.chmod(0o755)
    root = loose / "root"
    rec = FileRecorder(workspace_dir(root, workspace), workspace)
    rec.message(Message("user", "x"), None)
    rec.close()
    assert mode(loose) == 0o755
    assert mode(root) == 0o700


def test_an_existing_loose_root_is_tightened(tmp_path: Path, directory: Path, workspace: Path) -> None:
    directory.parent.mkdir()
    directory.parent.chmod(0o755)
    rec = FileRecorder(directory, workspace)
    rec.message(Message("user", "x"), None)
    rec.close()
    assert mode(directory.parent) == 0o700


def test_a_failed_root_chmod_is_retried_on_the_next_write(
    monkeypatch: pytest.MonkeyPatch, directory: Path, workspace: Path
) -> None:
    root = directory.parent
    root.mkdir()
    root.chmod(0o755)
    real_chmod = os.chmod
    failed = {"done": False}

    def flaky_chmod(path: Any, *args: Any, **kwargs: Any) -> None:
        if not failed["done"] and Path(path) == root:
            failed["done"] = True
            raise PermissionError(1, "not permitted")
        real_chmod(path, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", flaky_chmod)
    rec = FileRecorder(directory, workspace)
    with pytest.raises(SessionWriteError):
        rec.message(Message("user", "one"), None)
    assert mode(root) == 0o755
    rec.message(Message("user", "two"), None)
    rec.close()
    assert mode(root) == 0o700
    assert rec.path is not None
    assert [r["content"] for r in lines(rec.path) if r["type"] == "message"] == ["one", "two"]


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
