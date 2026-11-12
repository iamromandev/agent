"""The MemoryRecorder double behaves like FileRecorder where the loop tests depend on it."""

from __future__ import annotations

import pytest
from src.agent.provider.base import Message
from src.agent.session import SessionWriteError, ToolMeta

from tests.fakes import MemoryRecorder, RecordedCompaction


def compact(recorder: MemoryRecorder, history: list[Message], after: int = 10) -> None:
    recorder.compaction(history, {"c1": ToolMeta("read a.py", "1 lines")}, 1, 2, 100, after)


def test_a_failed_compaction_is_queued_and_written_by_the_next_successful_call() -> None:
    recorder = MemoryRecorder(fail=1)
    summary = [Message("user", "s")]

    with pytest.raises(SessionWriteError):
        compact(recorder, summary)
    assert recorder.compactions == []
    recorder.message(Message("user", "later"), None)

    assert recorder.compactions == [RecordedCompaction(summary, {"c1": ToolMeta("read a.py", "1 lines")}, 1, 2, 100, 10)]
    assert [m.content for m, _ in recorder.messages] == ["later"]


def test_queued_messages_and_compactions_are_written_in_the_order_they_came() -> None:
    recorder = MemoryRecorder(fail=2)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)
    with pytest.raises(SessionWriteError):
        compact(recorder, [Message("user", "s")])

    recorder.message(Message("user", "b"), None)

    kinds = ["message" if isinstance(r, tuple) else "compaction" for r in recorder.records]
    assert kinds == ["message", "compaction", "message"]


def test_a_queued_compaction_keeps_the_history_it_had_when_it_was_recorded() -> None:
    recorder = MemoryRecorder(fail=1)
    history = [Message("user", "s")]
    with pytest.raises(SessionWriteError):
        compact(recorder, history)

    history[0].content = "changed later"
    recorder.message(Message("user", "b"), None)

    assert recorder.compactions[0].history[0].content == "s"


def test_close_writes_what_is_still_queued() -> None:
    recorder = MemoryRecorder(fail=1)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)

    recorder.close()

    assert [m.content for m, _ in recorder.messages] == ["a"]
    assert recorder.closed == 1


def test_close_makes_a_final_attempt_for_a_queued_compaction_too() -> None:
    recorder = MemoryRecorder(fail=1)
    with pytest.raises(SessionWriteError):
        compact(recorder, [Message("user", "s")])

    recorder.close()

    assert len(recorder.compactions) == 1


def test_close_never_raises_when_the_final_attempt_fails() -> None:
    recorder = MemoryRecorder(fail=5)
    with pytest.raises(SessionWriteError):
        recorder.message(Message("user", "a"), None)

    recorder.close()  # the second failure is swallowed

    assert recorder.messages == []
    assert recorder.closed == 1


def test_close_with_nothing_queued_makes_no_write_attempt() -> None:
    recorder = MemoryRecorder()

    recorder.close()

    assert recorder.calls == 0
    assert recorder.closed == 1
