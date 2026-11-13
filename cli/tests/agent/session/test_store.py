from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from src.agent.provider.base import Message, ToolCall
from src.agent.session.paths import workspace_dir
from src.agent.session.recorder import FileRecorder
from src.agent.session.records import SessionHeader, ToolMeta, dumps, message_to_obj, title_from
from src.agent.session.store import (
    AmbiguousSession,
    LoadedSession,
    SessionError,
    SessionInfo,
    SessionNotFound,
    SessionStore,
    TranscriptItem,
    _updated_at,
    compaction_notice,
)

CLI_DIR = Path(__file__).resolve().parents[3]
STAMP = "2026-11-12T15:03:18Z"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "root"


@pytest.fixture
def store(root: Path, workspace: Path) -> SessionStore:
    return SessionStore(root, workspace)


def msg_obj(message: Message, meta: ToolMeta | None = None) -> dict[str, Any]:
    return {**message_to_obj(message, meta), "at": STAMP}


def write_session(
    root: Path,
    workspace: Path,
    session_id: str,
    records: Sequence[dict[str, Any]] = (),
    *,
    title: str | None = None,
    mtime: float | None = None,
    header: dict[str, Any] | None = None,
) -> Path:
    """Write a session file by hand, so ids and contents are exact."""
    directory = workspace_dir(root, workspace)
    directory.mkdir(parents=True, exist_ok=True)
    header_obj = header or SessionHeader(session_id, str(workspace.resolve()), STAMP).to_obj()
    objs = [header_obj]
    if title is not None:
        objs.append({"type": "title", "title": title, "at": STAMP})
    objs.extend(records)
    path = directory / f"{session_id}.jsonl"
    path.write_text("".join(dumps(o) + "\n" for o in objs))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def user(text: str) -> dict[str, Any]:
    return msg_obj(Message("user", text))


def assistant(text: str, calls: list[ToolCall] | None = None) -> dict[str, Any]:
    return msg_obj(Message("assistant", text, tool_calls=calls or []))


def tool(call_id: str, name: str, content: str, meta: ToolMeta | None = None) -> dict[str, Any]:
    return msg_obj(Message("tool", content, tool_call_id=call_id, name=name), meta)


def compaction(history: list[dict[str, Any]], pruned: int = 1, summarized: int = 2) -> dict[str, Any]:
    return {
        "type": "compaction",
        "history": history,
        "pruned": pruned,
        "summarized": summarized,
        "before": 900,
        "after": 400,
        "at": STAMP,
    }


def embedded(message: Message, meta: ToolMeta | None = None) -> dict[str, Any]:
    return message_to_obj(message, meta, embedded=True)


# --- list -------------------------------------------------------------------------------------


def test_list_of_a_missing_directory_is_empty(store: SessionStore) -> None:
    assert store.list() == []
    assert store.skipped == []
    assert store.latest() is None


def test_list_is_newest_updated_first(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("old")], title="old", mtime=1_000)
    write_session(root, workspace, "01CCCCCCCCCCCCCCCCCCCCCCCC", [user("new")], title="new", mtime=3_000)
    write_session(root, workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", [user("mid")], title="mid", mtime=2_000)
    assert [info.title for info in store.list()] == ["new", "mid", "old"]


def test_list_reports_id_path_times_title_and_counts(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(
        root,
        workspace,
        "01JF3KAAAAAAAAAAAAAAAAAAAA",
        [
            user("hi"),
            assistant("", [ToolCall("c1", "read", {"path": "a"})]),
            tool("c1", "read", "x", ToolMeta("read a", "1 lines")),
            assistant("done"),
            user("again"),
        ],
        title="hi",
        mtime=1_700_000_000,
    )
    [info] = store.list()
    assert info == SessionInfo(
        id="01JF3KAAAAAAAAAAAAAAAAAAAA",
        path=path,
        title="hi",
        created_at=datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC),
        updated_at=datetime.fromtimestamp(1_700_000_000, UTC),
        message_count=4,  # user + assistant records, not tool results
    )
    assert info.updated_at.tzinfo is not None


def test_a_session_without_a_title_record_is_untitled(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [assistant("hello")])
    assert [(i.title, i.message_count) for i in store.list()] == [("(untitled)", 1)]


def test_the_last_title_record_wins(store: SessionStore, root: Path, workspace: Path) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a")], title="first")
    with path.open("a") as f:
        f.write(dumps({"type": "title", "title": "second", "at": STAMP}) + "\n")
    assert store.list()[0].title == "second"


def test_message_text_that_looks_like_a_record_is_not_counted(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    sneaky = 'x\n{"type":"message","role":"user","content":"fake"}\n{"type":"title","title":"fake"}'
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user(sneaky), assistant(sneaky)], title="real")
    [info] = store.list()
    assert (info.title, info.message_count) == ("real", 2)


def test_a_cut_last_line_is_not_counted_or_used_as_a_title(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a")], title="real")
    with path.open("a") as f:
        f.write('{"type":"title","title":"cut off')
        f.write("\n" + '{"type":"message","role":"user","content":"cut')
    [info] = store.list()
    assert (info.title, info.message_count) == ("real", 1)


def test_list_does_not_parse_every_line(
    monkeypatch: pytest.MonkeyPatch, store: SessionStore, root: Path, workspace: Path
) -> None:
    records = [user(f"q{i}") if i % 2 == 0 else assistant(f"a{i}") for i in range(500)]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records, title="t")
    calls = []
    real_loads = json.loads

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", counting)
    [info] = store.list()
    assert info.message_count == 500
    assert len(calls) <= 3  # the header and the title; never the 500 messages


@pytest.mark.parametrize(
    "first_line",
    [
        "this is not json",
        "",
        "[]",
        '{"type":"message","role":"user","content":"no header"}',
        '{"type":"session","id":"x"}',
        '{"type":"session","version":1,"id":"x","workspace":"/w","created_at":"not a date"}',
    ],
)
def test_a_bad_header_is_skipped_and_reported(
    store: SessionStore, root: Path, workspace: Path, first_line: str
) -> None:
    good = write_session(root, workspace, "01GGGGGGGGGGGGGGGGGGGGGGGG", [user("ok")], title="ok")
    bad = workspace_dir(root, workspace) / "01BBBBBBBBBBBBBBBBBBBBBBBB.jsonl"
    bad.write_text(first_line + "\n" + dumps(user("x")) + "\n")
    assert [i.path for i in store.list()] == [good]
    assert store.skipped == [bad]


def test_an_empty_file_is_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01GGGGGGGGGGGGGGGGGGGGGGGG", [user("ok")])
    empty = workspace_dir(root, workspace) / "01EEEEEEEEEEEEEEEEEEEEEEEE.jsonl"
    empty.write_text("")
    assert len(store.list()) == 1
    assert store.skipped == [empty]


def test_a_header_cut_in_the_middle_is_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    cut = workspace_dir(root, workspace)
    cut.mkdir(parents=True)
    path = cut / "01CCCCCCCCCCCCCCCCCCCCCCCC.jsonl"
    path.write_text('{"type":"session","version":1,"id":"01CCCC')
    assert store.list() == []
    assert store.skipped == [path]


def test_version_2_is_skipped_and_reported(store: SessionStore, root: Path, workspace: Path) -> None:
    future = SessionHeader("01VVVVVVVVVVVVVVVVVVVVVVVV", str(workspace.resolve()), STAMP, version=2).to_obj()
    path = write_session(root, workspace, "01VVVVVVVVVVVVVVVVVVVVVVVV", [user("x")], header=future)
    assert store.list() == []
    assert store.skipped == [path]


def test_a_foreign_workspace_is_skipped_and_reported(
    store: SessionStore, root: Path, workspace: Path, tmp_path: Path
) -> None:
    other = SessionHeader("01FFFFFFFFFFFFFFFFFFFFFFFF", str((tmp_path / "other").resolve()), STAMP).to_obj()
    path = write_session(root, workspace, "01FFFFFFFFFFFFFFFFFFFFFFFF", [user("x")], header=other)
    assert store.list() == []
    assert store.skipped == [path]


def test_skipped_is_reset_by_the_next_list(store: SessionStore, root: Path, workspace: Path) -> None:
    bad = write_session(root, workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", [user("x")], header={"type": "nope"})
    store.list()
    assert store.skipped == [bad]
    bad.unlink()
    store.list()
    assert store.skipped == []


def test_files_that_are_not_sessions_are_ignored(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("x")])
    (workspace_dir(root, workspace) / "notes.txt").write_text("hi")
    (workspace_dir(root, workspace) / "sub.jsonl").mkdir()
    assert len(store.list()) == 1
    assert store.skipped == []


def test_other_workspaces_do_not_leak(store: SessionStore, root: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    write_session(root, other, "01OOOOOOOOOOOOOOOOOOOOOOOO", [user("x")])
    assert store.list() == []


def test_latest_is_the_most_recently_updated(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a")], mtime=5_000)
    write_session(root, workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", [user("b")], mtime=4_000)
    latest = store.latest()
    assert latest is not None and latest.id == "01AAAAAAAAAAAAAAAAAAAAAAAA"


# --- find -------------------------------------------------------------------------------------


@pytest.fixture
def two_sessions(root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01JF3KAAAAAAAAAAAAAAAAAAAA", [user("a")], mtime=1_000)
    write_session(root, workspace, "01JF3MBBBBBBBBBBBBBBBBBBBB", [user("b")], mtime=2_000)


@pytest.mark.usefixtures("two_sessions")
def test_find_by_a_six_character_prefix(store: SessionStore) -> None:
    assert store.find("01JF3K").id == "01JF3KAAAAAAAAAAAAAAAAAAAA"


@pytest.mark.usefixtures("two_sessions")
def test_find_by_the_full_id_and_case_insensitively(store: SessionStore) -> None:
    assert store.find("01JF3MBBBBBBBBBBBBBBBBBBBB").id == "01JF3MBBBBBBBBBBBBBBBBBBBB"
    assert store.find("01jf3m").id == "01JF3MBBBBBBBBBBBBBBBBBBBB"


@pytest.mark.usefixtures("two_sessions")
def test_find_with_a_short_prefix_names_the_id(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound, match="01JF3"):
        store.find("01JF3")


@pytest.mark.usefixtures("two_sessions")
def test_find_with_an_unknown_id_names_the_id(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound, match="ZZZZZZ"):
        store.find("ZZZZZZ")


def test_find_is_ambiguous_when_two_ids_share_a_six_character_prefix(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(root, workspace, "01JF3KAAAAAAAAAAAAAAAAAAAA", [user("a")])
    write_session(root, workspace, "01JF3KBBBBBBBBBBBBBBBBBBBB", [user("b")])
    with pytest.raises(AmbiguousSession, match="01JF3K"):
        store.find("01JF3K")
    assert store.find("01JF3KA").id == "01JF3KAAAAAAAAAAAAAAAAAAAA"


def test_the_session_errors_are_session_errors() -> None:
    assert issubclass(SessionNotFound, SessionError)
    assert issubclass(AmbiguousSession, SessionError)


# --- load -------------------------------------------------------------------------------------


def test_load_rebuilds_history_tool_meta_and_items(store: SessionStore, root: Path, workspace: Path) -> None:
    call = ToolCall("c1", "read", {"path": "a.py"})
    meta = ToolMeta("read a.py", "3 lines")
    write_session(
        root,
        workspace,
        "01AAAAAAAAAAAAAAAAAAAAAAAA",
        [
            user("show a.py"),
            assistant("", [call]),
            tool("c1", "read", "x = 1", meta),
            assistant("it sets x"),
            tool("c9", "bash", "boom"),
        ],
        title="show a.py",
    )
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert isinstance(loaded, LoadedSession)
    assert loaded.history == [
        Message("user", "show a.py"),
        Message("assistant", "", tool_calls=[call]),
        Message("tool", "x = 1", tool_call_id="c1", name="read"),
        Message("assistant", "it sets x"),
        Message("tool", "boom", tool_call_id="c9", name="bash"),
    ]
    assert loaded.tool_meta == {"c1": meta}
    assert loaded.items == [
        TranscriptItem("user", "show a.py"),
        TranscriptItem("tool", "read a.py", "3 lines"),
        TranscriptItem("assistant", "it sets x"),
        TranscriptItem("tool", "bash", ""),
    ]
    assert loaded.skipped_lines == 0
    assert loaded.info == store.list()[0]


def test_load_info_matches_the_listing_for_the_same_file(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a"), assistant("b")], title="t", mtime=9_000)
    assert store.load("01AAAAAAAAAAAAAAAAAAAAAAAA").info == store.list()[0]


def test_load_skips_a_cut_last_line_and_keeps_every_complete_record(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("one"), assistant("two"), user("three")])
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 25])  # cut in the middle of the last record
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one", "two"]
    assert loaded.skipped_lines == 0  # a cut last line is not counted


def test_load_skips_a_last_line_that_is_valid_json_but_has_no_newline(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("one")])
    with path.open("a") as f:
        f.write(dumps(user("unterminated")))
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one"]


def test_load_skips_garbage_and_invalid_records_in_the_middle(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("one")])
    with path.open("ab") as f:
        f.write(b"garbage that is not json\n")
        f.write(b"\xff\xfe\x00 not utf-8\n")
        f.write(b'{"type":"message","role":"robot","content":"bad role"}\n')
        f.write(b"[1,2,3]\n")
        f.write((dumps(assistant("two")) + "\n").encode())
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one", "two"]
    assert loaded.skipped_lines == 4


def test_load_ignores_blank_lines_and_unknown_record_types(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("one")])
    with path.open("a") as f:
        f.write('\n{"type":"from-the-future","x":1}\n')
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one"]
    assert loaded.skipped_lines == 0


def test_a_compaction_record_replaces_the_history_and_adds_a_notice(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    call = ToolCall("c1", "read", {"path": "a"})
    kept = [
        Message("user", "Summary of the earlier conversation:\nstuff"),
        Message("assistant", "", tool_calls=[call]),
        Message("tool", "[output pruned: read a · 1 lines]", tool_call_id="c1", name="read"),
        Message("user", "latest"),
    ]
    write_session(
        root,
        workspace,
        "01AAAAAAAAAAAAAAAAAAAAAAAA",
        [
            user("old question"),
            assistant("old answer"),
            compaction(
                [embedded(m, ToolMeta("read a", "1 lines") if m.role == "tool" else None) for m in kept],
                pruned=3,
                summarized=5,
            ),
            assistant("new answer"),
            user("follow-up"),
        ],
        title="old question",
    )
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert loaded.history == [*kept, Message("assistant", "new answer"), Message("user", "follow-up")]
    assert loaded.tool_meta == {"c1": ToolMeta("read a", "1 lines")}
    assert loaded.items == [
        TranscriptItem("user", "old question"),
        TranscriptItem("assistant", "old answer"),
        TranscriptItem("notice", "Context compacted: pruned 3 tool outputs, summarized 5 messages (ctx ~900 \u2192 ~400)"),
        TranscriptItem("assistant", "new answer"),
        TranscriptItem("user", "follow-up"),
    ]


def test_the_replayed_notice_leaves_out_summarized_when_nothing_was_summarized(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(
        root,
        workspace,
        "01AAAAAAAAAAAAAAAAAAAAAAAA",
        [user("q"), compaction([embedded(Message("user", "q"))], pruned=2, summarized=0)],
        title="q",
    )

    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert loaded.items[-1] == TranscriptItem("notice", "Context compacted: pruned 2 tool outputs (ctx ~900 \u2192 ~400)")


@pytest.mark.parametrize(
    ("pruned", "summarized", "text"),
    [
        (1, 0, "Context compacted: pruned 1 tool output"),
        (0, 0, "Context compacted: pruned 0 tool outputs"),
        (2, 0, "Context compacted: pruned 2 tool outputs"),
        (1, 1, "Context compacted: pruned 1 tool output, summarized 1 message"),
        (2, 2, "Context compacted: pruned 2 tool outputs, summarized 2 messages"),
    ],
)
def test_the_compaction_notice_uses_singular_and_plural_correctly(pruned: int, summarized: int, text: str) -> None:
    assert compaction_notice(pruned, summarized) == text
    assert compaction_notice(pruned, summarized, 900, 400) == f"{text} (ctx ~900 \u2192 ~400)"


def test_the_replayed_notice_leaves_out_the_context_part_when_the_record_has_no_sizes(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    record = compaction([embedded(Message("user", "q"))], pruned=1, summarized=2)
    del record["before"]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("q"), record], title="q")

    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert loaded.items[-1] == TranscriptItem("notice", "Context compacted: pruned 1 tool output, summarized 2 messages")


def test_a_recorded_session_loads_through_the_real_recorder_and_keeps_every_message_record(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    history = [Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2"), Message("assistant", "a2")]
    for m in history:
        recorder.message(m, None)
    compacted = [Message("user", "Summary of the earlier conversation:\ns"), history[2], history[3]]
    recorder.compaction(compacted, {}, 0, 2, 100, 50)
    recorder.message(Message("user", "q3"), None)
    recorder.message(Message("assistant", "a3"), None)
    recorder.close()

    assert recorder.path is not None
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [r["content"] for r in records if r["type"] == "message"] == ["q1", "a1", "q2", "a2", "q3", "a3"]

    loaded = store.load(store.list()[0].id)
    assert loaded.history == [*compacted, Message("user", "q3"), Message("assistant", "a3")]
    assert [i.text for i in loaded.items if i.kind != "notice"] == ["q1", "a1", "q2", "a2", "q3", "a3"]
    assert [i.kind for i in loaded.items].count("notice") == 1
    assert loaded.info.title == "q1"
    assert loaded.info.message_count == 6


def test_an_invalid_compaction_record_is_skipped_and_counted(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    bad = compaction([{"role": "robot", "content": "x"}])
    missing = {"type": "compaction", "history": [], "at": STAMP}
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("keep"), bad, missing, assistant("also")])
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["keep", "also"]
    assert loaded.skipped_lines == 2
    assert all(i.kind != "notice" for i in loaded.items)


def test_load_of_a_session_with_only_a_header_is_empty(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA")
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert (loaded.history, loaded.items, loaded.tool_meta) == ([], [], {})


def test_load_of_an_unknown_id_is_not_found(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("x")])
    with pytest.raises(SessionNotFound, match="01MISSING"):
        store.load("01MISSING")
    with pytest.raises(SessionNotFound):
        SessionStore(root / "nowhere", workspace).load("01AAAAAAAAAAAAAAAAAAAAAAAA")


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".", "..", "x\x00y"])
def test_load_refuses_ids_that_could_leave_the_session_folder(
    store: SessionStore, root: Path, workspace: Path, bad_id: str
) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("x")])
    (root / "escape.jsonl").write_text(dumps(SessionHeader("x", str(workspace), STAMP).to_obj()) + "\n")
    with pytest.raises(SessionNotFound):
        store.load(bad_id)


@pytest.mark.parametrize("first_line", ["not json", "", '{"type":"session"}'])
def test_load_refuses_a_bad_header(store: SessionStore, root: Path, workspace: Path, first_line: str) -> None:
    directory = workspace_dir(root, workspace)
    directory.mkdir(parents=True)
    (directory / "01BBBBBBBBBBBBBBBBBBBBBBBB.jsonl").write_text(first_line + "\n" + dumps(user("x")) + "\n")
    with pytest.raises(SessionError, match="01BBBBBBBBBBBBBBBBBBBBBBBB"):
        store.load("01BBBBBBBBBBBBBBBBBBBBBBBB")


def test_load_refuses_version_2(store: SessionStore, root: Path, workspace: Path) -> None:
    future = SessionHeader("01VVVVVVVVVVVVVVVVVVVVVVVV", str(workspace.resolve()), STAMP, version=2).to_obj()
    write_session(root, workspace, "01VVVVVVVVVVVVVVVVVVVVVVVV", [user("x")], header=future)
    with pytest.raises(SessionError, match="version"):
        store.load("01VVVVVVVVVVVVVVVVVVVVVVVV")


def test_load_refuses_a_foreign_workspace(store: SessionStore, root: Path, workspace: Path, tmp_path: Path) -> None:
    other = SessionHeader("01FFFFFFFFFFFFFFFFFFFFFFFF", str((tmp_path / "other").resolve()), STAMP).to_obj()
    write_session(root, workspace, "01FFFFFFFFFFFFFFFFFFFFFFFF", [user("x")], header=other)
    with pytest.raises(SessionError, match="workspace"):
        store.load("01FFFFFFFFFFFFFFFFFFFFFFFF")


def test_a_refused_load_is_a_plain_session_error_not_not_found(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(root, workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", [user("x")], header={"type": "nope"})
    with pytest.raises(SessionError) as excinfo:
        store.load("01BBBBBBBBBBBBBBBBBBBBBBBB")
    assert not isinstance(excinfo.value, SessionNotFound)


# --- crash safety (NFR1) ----------------------------------------------------------------------

CHILD = """
import sys
from pathlib import Path
from src.agent.provider.base import Message
from src.agent.session.recorder import FileRecorder

recorder = FileRecorder(Path(sys.argv[1]), Path(sys.argv[2]))
payload = "x" * 20_000
count = 0
while True:
    recorder.message(Message("user" if count % 2 == 0 else "assistant", f"{count} {payload}"), None)
    count += 1
    if count == 1:
        print("written", flush=True)
"""


@pytest.mark.parametrize("delay", [0.1, 0.1, 0.15])
def test_a_child_killed_with_sigkill_while_appending_leaves_a_loadable_session(
    store: SessionStore, root: Path, workspace: Path, tmp_path: Path, delay: float
) -> None:
    directory = workspace_dir(root, workspace)
    env = {**os.environ, "PYTHONPATH": str(CLI_DIR)}
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(directory), str(workspace)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "written", child.stderr.read() if child.stderr else ""
        time.sleep(delay)
    finally:
        child.send_signal(signal.SIGKILL)  # only the child this test started
        child.wait(timeout=10)
        if child.stdout:
            child.stdout.close()
        if child.stderr:
            child.stderr.close()

    [info] = store.list()
    loaded = store.load(info.id)
    assert len(loaded.history) >= 1
    assert loaded.skipped_lines == 0
    assert [m.role for m in loaded.history][:2] == ["user", "assistant"][: len(loaded.history)]
    assert info.message_count == len(loaded.history)


# --- lone surrogates and non-UTF-8 workspace paths ----------------------------------------------

LONE = "bad \ud800 text \udfff end"


def record_and_load(root: Path, workspace: Path, messages: list[Message]) -> LoadedSession:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    try:
        for message in messages:
            recorder.message(message, None)
    finally:
        recorder.close()
    [info] = SessionStore(root, workspace).list()
    return SessionStore(root, workspace).load(info.id)


def test_a_lone_surrogate_in_user_content_and_title_round_trips(root: Path, workspace: Path) -> None:
    loaded = record_and_load(root, workspace, [Message("user", LONE)])
    assert [m.content for m in loaded.history] == [LONE]
    assert loaded.info.title == title_from(LONE)
    assert "\ud800" in loaded.info.title
    assert loaded.skipped_lines == 0


def test_a_lone_surrogate_in_assistant_content_round_trips(root: Path, workspace: Path) -> None:
    loaded = record_and_load(root, workspace, [Message("user", "hi"), Message("assistant", LONE)])
    assert [m.content for m in loaded.history] == ["hi", LONE]
    assert loaded.skipped_lines == 0


def test_a_lone_surrogate_in_tool_arguments_round_trips(root: Path, workspace: Path) -> None:
    call = ToolCall("c1", "write", {"path": "a\ud800b", "content": LONE})
    loaded = record_and_load(root, workspace, [Message("user", "hi"), Message("assistant", "", tool_calls=[call])])
    assert loaded.history[1].tool_calls == [call]
    assert loaded.skipped_lines == 0


def test_a_lone_surrogate_in_tool_output_round_trips(root: Path, workspace: Path) -> None:
    tool_message = Message("tool", LONE, tool_call_id="c1", name="read")
    loaded = record_and_load(root, workspace, [Message("user", "hi"), tool_message])
    assert loaded.history[1].content == LONE
    assert loaded.skipped_lines == 0


def test_a_record_with_a_surrogate_does_not_damage_its_neighbours(root: Path, workspace: Path) -> None:
    loaded = record_and_load(root, workspace, [Message("user", "é日本"), Message("assistant", LONE), Message("user", "ü")])
    assert [m.content for m in loaded.history] == ["é日本", LONE, "ü"]


def test_a_workspace_path_with_a_non_utf8_byte_round_trips(root: Path, tmp_path: Path) -> None:
    workspace = tmp_path / os.fsdecode(b"x\xff")  # the name is never created: only the path is used
    loaded = record_and_load(root, workspace, [Message("user", "hi")])
    assert [m.content for m in loaded.history] == ["hi"]
    assert SessionStore(root, workspace).list() != []


# --- resuming after a cut tail ------------------------------------------------------------------


def started_session(root: Path, workspace: Path, messages: list[Message]) -> Path:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    for message in messages:
        recorder.message(message, None)
    recorder.close()
    assert recorder.path is not None
    return recorder.path


def test_resume_after_a_cut_tail_keeps_the_new_message(root: Path, workspace: Path) -> None:
    path = started_session(root, workspace, [Message("user", "first"), Message("assistant", "second")])
    path.write_bytes(path.read_bytes()[:-15])  # the last record is cut, with no newline

    resumed = FileRecorder.resume(path, workspace)
    resumed.message(Message("user", "third"), None)
    resumed.close()

    loaded = SessionStore(root, workspace).load(path.stem)
    assert [m.content for m in loaded.history] == ["first", "third"]
    assert loaded.skipped_lines <= 1


def test_resume_after_a_complete_last_record_without_a_newline_keeps_both(root: Path, workspace: Path) -> None:
    path = started_session(root, workspace, [Message("user", "first"), Message("assistant", "second")])
    path.write_bytes(path.read_bytes()[:-1])  # only the final "\n" is missing

    resumed = FileRecorder.resume(path, workspace)
    resumed.message(Message("user", "third"), None)
    resumed.close()

    loaded = SessionStore(root, workspace).load(path.stem)
    assert [m.content for m in loaded.history] == ["first", "second", "third"]
    assert loaded.skipped_lines == 0


# --- deeply nested JSON and absurd timestamps ---------------------------------------------------

DEEP = b"[" * 200_000
DEEP_ID = "01DDDDDDDDDDDDDDDDDDDDDDDD"


def test_a_deeply_nested_header_is_skipped_and_reported(store: SessionStore, root: Path, workspace: Path) -> None:
    path = write_session(root, workspace, DEEP_ID, [user("x")])
    path.write_bytes(DEEP + b"\n" + b"\n".join(path.read_bytes().split(b"\n")[1:]))
    assert store.list() == []
    assert store.skipped == [path]
    with pytest.raises(SessionError, match="header"):
        store.load(DEEP_ID)


def test_a_deeply_nested_message_line_is_counted_as_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    path = write_session(root, workspace, DEEP_ID, [user("before")])
    with path.open("ab") as f:
        f.write(DEEP + b"\n")
        f.write((dumps(assistant("after")) + "\n").encode())
    [info] = store.list()
    loaded = store.load(info.id)
    assert [m.content for m in loaded.history] == ["before", "after"]
    assert loaded.skipped_lines == 1


def test_a_deeply_nested_title_line_is_ignored(store: SessionStore, root: Path, workspace: Path) -> None:
    path = write_session(root, workspace, DEEP_ID, [user("x")], title="real")
    with path.open("ab") as f:
        f.write(b'{"type":"title","title":' + DEEP + b"\n")
    [info] = store.list()
    assert info.title == "real"


def test_an_absurd_mtime_falls_back_to_the_created_time(store: SessionStore, root: Path, workspace: Path) -> None:
    path = write_session(root, workspace, DEEP_ID, [user("x")])
    try:
        os.utime(path, (2**60, 2**60))
    except (OverflowError, OSError):
        pytest.skip("this filesystem rejects the timestamp")
    try:
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except (OverflowError, ValueError, OSError):
        pass
    else:
        pytest.skip("this filesystem clamps the timestamp to a valid date")
    [info] = store.list()
    assert info.updated_at == datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC)
    assert store.load(info.id).info.updated_at == info.updated_at


@pytest.mark.parametrize("mtime", [1e30, -1e30, float("inf"), float("nan"), 2**70])
def test_updated_at_falls_back_when_the_mtime_is_not_a_date(mtime: float) -> None:
    fallback = datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC)
    assert _updated_at(mtime, fallback) == fallback


def test_updated_at_uses_the_mtime_when_it_is_a_date() -> None:
    fallback = datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC)
    assert _updated_at(1_700_000_000, fallback) == datetime.fromtimestamp(1_700_000_000, UTC)


# --- the id is the file name --------------------------------------------------------------------

ORIGINAL_ID = "01OOOOOOOOOOOOOOOOOOOOOOOO"
COPY_ID = "01CCCCCCCCCCCCCCCCCCCCCCCC"


def test_a_copied_file_is_listed_once_and_the_copy_is_skipped(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    original = write_session(root, workspace, ORIGINAL_ID, [user("x")], title="t")
    copy = original.with_name(f"{COPY_ID}.jsonl")
    shutil.copyfile(original, copy)
    assert [info.id for info in store.list()] == [ORIGINAL_ID]
    assert store.skipped == [copy]


def test_a_renamed_file_is_skipped_not_listed_under_an_id_load_cannot_find(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    original = write_session(root, workspace, ORIGINAL_ID, [user("x")])
    renamed = original.rename(original.with_name("renamed.jsonl"))
    assert store.list() == []
    assert store.skipped == [renamed]


def test_load_refuses_a_copy_whose_header_names_another_id(store: SessionStore, root: Path, workspace: Path) -> None:
    original = write_session(root, workspace, ORIGINAL_ID, [user("x")])
    shutil.copyfile(original, original.with_name(f"{COPY_ID}.jsonl"))
    with pytest.raises(SessionError) as excinfo:
        store.load(COPY_ID)
    assert not isinstance(excinfo.value, SessionNotFound)
    assert store.load(ORIGINAL_ID).info.id == ORIGINAL_ID


def test_a_file_name_that_is_not_a_valid_id_is_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    original = write_session(root, workspace, ORIGINAL_ID, [user("x")])
    odd = original.with_name("two words.jsonl")
    shutil.copyfile(original, odd)
    assert [info.id for info in store.list()] == [ORIGINAL_ID]
    assert store.skipped == [odd]


def test_the_listed_id_always_loads(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, ORIGINAL_ID, [user("x")])
    write_session(root, workspace, "01BBBBBBBBBBBBBBBBBBBBBBBB", [user("y")])
    for info in store.list():
        assert store.load(info.id).info == info


# --- symlinks and special files -----------------------------------------------------------------

LINK_ID = "01LLLLLLLLLLLLLLLLLLLLLLLL"


def symlinked_session(root: Path, workspace: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A valid session file kept elsewhere, with a symlink to it where the store looks."""
    real = write_session(root, workspace, LINK_ID, [user("x")], title="t")
    target = real.rename(tmp_path / "elsewhere.jsonl")
    real.symlink_to(target)
    return real, target


def test_list_skips_a_symlinked_session_file_and_reports_it(
    store: SessionStore, root: Path, workspace: Path, tmp_path: Path
) -> None:
    link, _ = symlinked_session(root, workspace, tmp_path)
    assert store.list() == []
    assert store.skipped == [link]


def test_load_refuses_a_symlinked_session_file(
    store: SessionStore, root: Path, workspace: Path, tmp_path: Path
) -> None:
    symlinked_session(root, workspace, tmp_path)
    with pytest.raises(SessionError):
        store.load(LINK_ID)


def test_list_and_load_skip_a_fifo_without_blocking(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    directory = workspace_dir(root, workspace)
    directory.mkdir(parents=True)
    fifo = directory / "x.jsonl"
    os.mkfifo(fifo)
    write_session(root, workspace, ORIGINAL_ID, [user("x")])
    assert [info.id for info in store.list()] == [ORIGINAL_ID]
    assert store.skipped == [fifo]
    os.mkfifo(directory / f"{LINK_ID}.jsonl")
    with pytest.raises(SessionError):
        store.load(LINK_ID)



def test_load_refuses_a_directory_named_like_a_session(store: SessionStore, root: Path, workspace: Path) -> None:
    (workspace_dir(root, workspace) / f"{LINK_ID}.jsonl").mkdir(parents=True)
    with pytest.raises(SessionError):
        store.load(LINK_ID)


# --- undo records ------------------------------------------------------------------------------


def undo_record(keep: object) -> dict[str, Any]:
    return {"type": "undo", "keep": keep, "at": STAMP}


def three_turns() -> list[dict[str, Any]]:
    return [user("one"), assistant("a1"), user("two"), assistant("a2"), user("three"), assistant("a3")]


def test_an_undo_record_drops_the_last_turn_from_history_and_items(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [*three_turns(), undo_record(4)])
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one", "a1", "two", "a2"]
    assert [i.text for i in loaded.items] == ["one", "a1", "two", "a2"]
    assert loaded.skipped_lines == 0


def test_a_session_without_undo_records_loads_every_turn(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", three_turns())
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert len(loaded.history) == 6
    assert len(loaded.items) == 6


def test_two_undo_records_in_a_row_drop_two_turns(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [*three_turns(), undo_record(4), undo_record(2)])
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one", "a1"]
    assert [i.text for i in loaded.items] == ["one", "a1"]


def test_messages_after_an_undo_continue_from_the_cut(store: SessionStore, root: Path, workspace: Path) -> None:
    records = [*three_turns(), undo_record(4), user("three again"), assistant("a3b"), undo_record(4)]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["one", "a1", "two", "a2"]


def test_an_undo_drops_tool_messages_and_their_meta_and_items(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    call = ToolCall("c1", "write", {"path": "a"})
    records = [
        user("one"),
        assistant("", [call]),
        tool("c1", "write", "ok", ToolMeta("write a", "1 line")),
        assistant("done"),
        undo_record(0),
    ]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert loaded.history == []
    assert loaded.tool_meta == {}
    assert loaded.items == []


def test_an_undo_right_after_a_compaction_counts_from_the_compacted_history(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    kept = [Message("user", "Summary of the earlier conversation:\nstuff"), Message("user", "latest")]
    records = [
        user("old"),
        assistant("old answer"),
        compaction([embedded(m) for m in kept]),
        user("after"),
        assistant("after answer"),
        undo_record(2),  # the compacted history has two messages: the undone turn is the one after it
    ]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert loaded.history == kept
    assert [i.kind for i in loaded.items] == ["user", "assistant", "notice"]
    assert loaded.skipped_lines == 0


@pytest.mark.parametrize("keep", [-1, 7, 1.5, "2", None, True, 0])
def test_a_bad_undo_record_is_skipped_and_counted(
    store: SessionStore, root: Path, workspace: Path, keep: object
) -> None:
    kept = [Message("user", "Summary:\nstuff"), Message("user", "latest")]
    records = [user("old"), compaction([embedded(m) for m in kept]), user("after"), undo_record(keep)]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["Summary:\nstuff", "latest", "after"]
    assert loaded.skipped_lines == 1


def test_an_undo_with_no_keep_is_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("one"), {"type": "undo", "at": STAMP}])
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert len(loaded.history) == 1
    assert loaded.skipped_lines == 1
