from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from src.engine.provider.base import Message, ToolCall
from src.engine.session.paths import workspace_dir
from src.engine.session.recorder import FileRecorder
from src.engine.session.records import SessionHeader, ToolMeta, dumps, message_to_obj
from src.engine.session.store import (
    AmbiguousSession,
    LoadedSession,
    SessionError,
    SessionNotFound,
    SessionStore,
    _updated_at,
    compaction_notice,
)
from src.engine.usage import UsageRecord

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


def test_a_cut_last_line_is_not_counted_or_used_as_a_title(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    path = write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a")], title="real")
    with path.open("a") as f:
        f.write('{"type":"title","title":"cut off')
        f.write("\n" + '{"type":"message","role":"user","content":"cut')
    [info] = store.list()
    assert (info.title, info.message_count) == ("real", 1)


@pytest.mark.parametrize(
    "first_line",
    [
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


def test_a_header_cut_in_the_middle_is_skipped(store: SessionStore, root: Path, workspace: Path) -> None:
    cut = workspace_dir(root, workspace)
    cut.mkdir(parents=True)
    path = cut / "01CCCCCCCCCCCCCCCCCCCCCCCC.jsonl"
    path.write_text('{"type":"session","version":1,"id":"01CCCC')
    assert store.list() == []
    assert store.skipped == [path]


# --- find -------------------------------------------------------------------------------------


@pytest.fixture
def two_sessions(root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01JF3KAAAAAAAAAAAAAAAAAAAA", [user("a")], mtime=1_000)
    write_session(root, workspace, "01JF3MBBBBBBBBBBBBBBBBBBBB", [user("b")], mtime=2_000)


@pytest.mark.usefixtures("two_sessions")
def test_find_with_a_short_prefix_names_the_id(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound, match="01JF3"):
        store.find("01JF3")


def test_find_is_ambiguous_when_two_ids_share_a_six_character_prefix(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(root, workspace, "01JF3KAAAAAAAAAAAAAAAAAAAA", [user("a")])
    write_session(root, workspace, "01JF3KBBBBBBBBBBBBBBBBBBBB", [user("b")])
    with pytest.raises(AmbiguousSession, match="01JF3K"):
        store.find("01JF3K")
    assert store.find("01JF3KA").id == "01JF3KAAAAAAAAAAAAAAAAAAAA"


# --- load -------------------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("pruned", "summarized", "text"),
    [
        (2, 0, "Context compacted: pruned 2 tool outputs"),
    ],
)
def test_the_compaction_notice_uses_singular_and_plural_correctly(pruned: int, summarized: int, text: str) -> None:
    assert compaction_notice(pruned, summarized) == text
    assert compaction_notice(pruned, summarized, 900, 400) == f"{text} (ctx ~900 \u2192 ~400)"


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


def test_load_of_an_unknown_id_is_not_found(store: SessionStore, root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("x")])
    with pytest.raises(SessionNotFound, match="01MISSING"):
        store.load("01MISSING")
    with pytest.raises(SessionNotFound):
        SessionStore(root / "nowhere", workspace).load("01AAAAAAAAAAAAAAAAAAAAAAAA")


@pytest.mark.parametrize("bad_id", ["x\x00y"])
def test_load_refuses_ids_that_could_leave_the_session_folder(
    store: SessionStore, root: Path, workspace: Path, bad_id: str
) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("x")])
    (root / "escape.jsonl").write_text(dumps(SessionHeader("x", str(workspace), STAMP).to_obj()) + "\n")
    with pytest.raises(SessionNotFound):
        store.load(bad_id)


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


# --- crash safety (NFR1) ----------------------------------------------------------------------

CHILD = """
import sys
from pathlib import Path
from src.engine.provider.base import Message
from src.engine.session.recorder import FileRecorder

recorder = FileRecorder(Path(sys.argv[1]), Path(sys.argv[2]))
payload = "x" * 20_000
count = 0
while True:
    recorder.message(Message("user" if count % 2 == 0 else "assistant", f"{count} {payload}"), None)
    count += 1
    if count == 1:
        print("written", flush=True)
"""


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


# --- resuming after a cut tail ------------------------------------------------------------------


def started_session(root: Path, workspace: Path, messages: list[Message]) -> Path:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    for message in messages:
        recorder.message(message, None)
    recorder.close()
    assert recorder.path is not None
    return recorder.path


# --- deeply nested JSON and absurd timestamps ---------------------------------------------------

DEEP = b"[" * 200_000
DEEP_ID = "01DDDDDDDDDDDDDDDDDDDDDDDD"


@pytest.mark.parametrize("mtime", [float("nan")])
def test_updated_at_falls_back_when_the_mtime_is_not_a_date(mtime: float) -> None:
    fallback = datetime(2026, 11, 12, 15, 3, 18, tzinfo=UTC)
    assert _updated_at(mtime, fallback) == fallback


# --- the id is the file name --------------------------------------------------------------------

ORIGINAL_ID = "01OOOOOOOOOOOOOOOOOOOOOOOO"
COPY_ID = "01CCCCCCCCCCCCCCCCCCCCCCCC"


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


# --- symlinks and special files -----------------------------------------------------------------

LINK_ID = "01LLLLLLLLLLLLLLLLLLLLLLLL"


def symlinked_session(root: Path, workspace: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A valid session file kept elsewhere, with a symlink to it where the store looks."""
    real = write_session(root, workspace, LINK_ID, [user("x")], title="t")
    target = real.rename(tmp_path / "elsewhere.jsonl")
    real.symlink_to(target)
    return real, target


def test_load_refuses_a_symlinked_session_file(
    store: SessionStore, root: Path, workspace: Path, tmp_path: Path
) -> None:
    symlinked_session(root, workspace, tmp_path)
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


@pytest.mark.parametrize("keep", [True])
def test_a_bad_undo_record_is_skipped_and_counted(
    store: SessionStore, root: Path, workspace: Path, keep: object
) -> None:
    kept = [Message("user", "Summary:\nstuff"), Message("user", "latest")]
    records = [user("old"), compaction([embedded(m) for m in kept]), user("after"), undo_record(keep)]
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["Summary:\nstuff", "latest", "after"]
    assert loaded.skipped_lines == 1


# --- todo list ---------------------------------------------------------------------------------


def todo_turn(call_id: str, *pairs: tuple[str, str], ok: bool = True) -> list[dict[str, Any]]:
    items = [{"content": c, "status": s} for c, s in pairs]
    records = [assistant("", [ToolCall(call_id, "todo", {"items": items})])]
    if ok:
        records.append(tool(call_id, "todo", "Updated", ToolMeta("todo", "x")))
    return records


# --- model switches ---------------------------------------------------------------------------


def model_record(name: str, window: int) -> dict[str, Any]:
    return {"type": "model", "at": STAMP, "model": name, "window": window}


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "model", "at": STAMP, "model": "m", "window": True},
    ],
)
def test_a_malformed_model_record_is_skipped_like_any_bad_record(
    store: SessionStore, root: Path, workspace: Path, bad: dict[str, Any]
) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("q"), bad])

    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert [i.text for i in loaded.items] == ["q"]
    assert loaded.skipped_lines == 1


def test_an_undo_that_cuts_back_past_a_model_notice_removes_it(
    store: SessionStore, root: Path, workspace: Path
) -> None:
    write_session(
        root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA",
        [user("q"), assistant("a"), model_record("m", 4096), user("q2"), assistant("a2"), {"type": "undo", "keep": 0, "at": STAMP}],
    )

    loaded = store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")

    assert loaded.items == []
    assert loaded.history == []


# --- usage records -----------------------------------------------------------------------------


def usage_obj(model: str = "m", i: int = 100, o: int = 20, ms: int = 1000, compaction: bool = False) -> dict[str, Any]:
    return UsageRecord(model, i, o, ms, compaction, STAMP).to_obj()


def load_records(store: SessionStore, root: Path, workspace: Path, records: list[dict[str, Any]]) -> LoadedSession:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", records)
    return store.load("01AAAAAAAAAAAAAAAAAAAAAAAA")


@pytest.mark.parametrize("patch", [{"output_tokens": "5"}, {"model": None}])
def test_a_malformed_usage_record_is_skipped_and_counted(
    store: SessionStore, root: Path, workspace: Path, patch: dict[str, Any]
) -> None:
    loaded = load_records(store, root, workspace, [user("q"), usage_obj(), {**usage_obj(), **patch}])
    assert loaded.skipped_lines == 1 and loaded.usage.calls == 1


def test_undo_and_compaction_keep_the_totals(store: SessionStore, root: Path, workspace: Path) -> None:
    records = [*three_turns(), usage_obj(), undo_record(4), compaction([user("s")]), usage_obj(compaction=True)]
    loaded = load_records(store, root, workspace, records)
    assert loaded.usage.calls == 2 and loaded.usage.total.compactions == 1


def new_header(workspace: Path) -> dict[str, Any]:
    return SessionHeader("01AAAAAAAAAAAAAAAAAAAAAAAA", str(workspace.resolve()), STAMP, tracks_usage=True).to_obj()


# --- images -----------------------------------------------------------------------------------

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"data"


def test_images_survive_a_save_and_a_resume(root: Path, workspace: Path) -> None:
    from src.engine.provider.base import Image

    image = Image("png", PNG_BYTES, "a.png")
    loaded = record_and_load(root, workspace, [Message("user", "see @a.png", images=(image,)), Message("assistant", "ok")])
    assert loaded.history[0].images == (image,)
    assert loaded.history[1].images == ()
    assert loaded.skipped_lines == 0


def test_a_session_with_images_still_counts_its_messages(root: Path, workspace: Path) -> None:
    from src.engine.provider.base import Image

    path = started_session(root, workspace, [Message("user", "x", images=(Image("png", PNG_BYTES, "a"),))])
    assert path.read_bytes().count(b'\n{"type":"message","role":"user"') == 1
    [info] = SessionStore(root, workspace).list()
    assert info.message_count == 1


def test_a_record_with_bad_images_is_skipped_and_counted(root: Path, workspace: Path) -> None:
    bad = {**user("bad"), "images": [{"format": "bmp", "name": "a", "data": "AAAA"}]}
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a"), bad, assistant("b")])
    loaded = SessionStore(root, workspace).load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.content for m in loaded.history] == ["a", "b"]
    assert loaded.skipped_lines == 1


def test_replay_shows_one_notice_line_per_image_after_its_message(root: Path, workspace: Path) -> None:
    from src.engine.provider.base import Image

    images = (Image("png", PNG_BYTES, "a\x1b[2J.png"), Image("jpeg", b"\xff\xd8\xffx", "clipboard"))
    loaded = record_and_load(root, workspace, [Message("user", "look", images=images), Message("assistant", "ok")])
    assert [(i.kind, i.text) for i in loaded.items] == [
        ("user", "look"),
        ("notice", "image a\\x1b[2J.png (PNG, 1 KB)"),
        ("notice", "image clipboard (JPEG, 1 KB)"),
        ("assistant", "ok"),
    ]


def test_replay_shows_a_tool_image_line_after_the_tool_item(root: Path, workspace: Path) -> None:
    from src.engine.provider.base import Image

    message = Message("tool", "image a.png (PNG, 1 KB)", tool_call_id="c", name="read", images=(Image("png", PNG_BYTES, "a.png"),))
    loaded = record_and_load(root, workspace, [message])
    assert [i.kind for i in loaded.items] == ["tool", "notice"]


def test_an_old_file_without_images_loads_unchanged(root: Path, workspace: Path) -> None:
    write_session(root, workspace, "01AAAAAAAAAAAAAAAAAAAAAAAA", [user("a"), assistant("b")])
    loaded = SessionStore(root, workspace).load("01AAAAAAAAAAAAAAAAAAAAAAAA")
    assert [m.images for m in loaded.history] == [(), ()]
    assert [i.kind for i in loaded.items] == ["user", "assistant"]
