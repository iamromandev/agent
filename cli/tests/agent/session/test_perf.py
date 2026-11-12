"""NFR2 / AC17: saving, listing and resuming stay fast. Generated sessions, real files, wall-clock limits."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest
from src.agent.provider.base import Message, ToolCall
from src.agent.session import FileRecorder, SessionStore, ToolMeta, workspace_dir

APPEND_MEDIAN_SECONDS = 0.005  # a message adds under 5 ms at the median
LIST_SECONDS = 0.5  # the picker for 200 sessions
LOAD_SECONDS = 1.0  # resuming a 5 MB session


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path.resolve()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "root"


def _exchange(n: int, size: int) -> list[tuple[Message, ToolMeta | None]]:
    call = ToolCall(f"c{n}", "read", {"path": f"src/file{n}.py"})
    return [
        (Message("user", f"question {n} about the code"), None),
        (Message("assistant", "Let me look.", [call]), None),
        (Message("tool", ("x = 1  # line\n" * (size // 15 + 1))[:size], tool_call_id=f"c{n}", name="read"), ToolMeta(f"read src/file{n}.py", "lines")),
        (Message("assistant", f"answer {n}: " + "text " * 40), None),
    ]


def test_appending_1000_messages_takes_under_5_ms_at_the_median(root: Path, workspace: Path) -> None:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    messages = [m for n in range(250) for m in _exchange(n, 400)]
    assert len(messages) == 1000
    durations: list[float] = []

    for message, meta in messages:
        start = time.perf_counter()
        recorder.message(message, meta)
        durations.append(time.perf_counter() - start)
    recorder.close()

    median = statistics.median(durations)
    print(f"AC17 append median={median * 1000:.3f} ms max={max(durations) * 1000:.3f} ms")
    assert median < APPEND_MEDIAN_SECONDS
    assert recorder.path is not None
    assert len(recorder.path.read_bytes().splitlines()) == 1002  # header, title, 1000 messages


def test_listing_200_sessions_takes_under_half_a_second(root: Path, workspace: Path) -> None:
    directory = workspace_dir(root, workspace)
    for n in range(200):
        recorder = FileRecorder(directory, workspace)
        for i in range(5):  # 20 records and about 10 KB each: a realistic short session
            for message, meta in _exchange(n * 10 + i, 600):
                recorder.message(message, meta)
        recorder.close()
        assert recorder.path is not None
        os.utime(recorder.path, (1_700_000_000 + n, 1_700_000_000 + n))
    store = SessionStore(root, workspace)

    start = time.perf_counter()
    infos = store.list()
    seconds = time.perf_counter() - start

    print(f"AC17 list(200)={seconds * 1000:.1f} ms")
    assert len(infos) == 200
    assert all(info.message_count == 15 for info in infos)
    assert infos[0].updated_at > infos[-1].updated_at
    assert seconds < LIST_SECONDS


def test_loading_a_5_mb_session_takes_under_a_second(root: Path, workspace: Path) -> None:
    recorder = FileRecorder(workspace_dir(root, workspace), workspace)
    for n in range(500):
        for message, meta in _exchange(n, 10_000):  # mostly tool output: about 10 KB a turn
            recorder.message(message, meta)
    recorder.close()
    assert recorder.path is not None
    size = recorder.path.stat().st_size
    assert size >= 5_000_000
    store = SessionStore(root, workspace)
    (info,) = store.list()

    start = time.perf_counter()
    loaded = store.load(info.id)
    seconds = time.perf_counter() - start

    print(f"AC17 load({size / 1e6:.1f} MB)={seconds * 1000:.1f} ms")
    assert len(loaded.history) == 2000
    assert len(loaded.items) == 2000  # every message here has text or a tool entry
    assert seconds < LOAD_SECONDS
