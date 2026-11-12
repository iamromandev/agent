from __future__ import annotations

import json
from typing import Any

import pytest
from src.agent.provider.base import Message, ToolCall
from src.agent.session.records import (
    FORMAT_VERSION,
    RecordError,
    SessionHeader,
    ToolMeta,
    dumps,
    message_from_obj,
    message_to_obj,
    title_from,
)

AC3_TEXT = "  Add   a --verbose\nflag to health check and also update the docs please  "


def roundtrip(message: Message, meta: ToolMeta | None = None) -> tuple[Message, ToolMeta | None]:
    obj = json.loads(dumps(message_to_obj(message, meta)))
    return message_from_obj(obj)


def test_user_message_round_trips() -> None:
    assert roundtrip(Message("user", "hello")) == (Message("user", "hello"), None)


def test_assistant_message_with_tool_calls_round_trips() -> None:
    calls = [ToolCall("c1", "read", {"path": "a.py"}), ToolCall("c2", "bash", {"command": "ls", "n": 3})]
    message = Message("assistant", "looking", tool_calls=calls)
    assert roundtrip(message) == (message, None)


def test_tool_message_with_meta_round_trips() -> None:
    message = Message("tool", "x = 1", tool_call_id="c1", name="read")
    meta = ToolMeta("read src/main.py", "31 lines")
    assert roundtrip(message, meta) == (message, meta)


def test_tool_message_without_meta_round_trips() -> None:
    message = Message("tool", "boom", tool_call_id="c1", name="read")
    assert roundtrip(message) == (message, None)


def test_unicode_and_newlines_round_trip() -> None:
    message = Message("user", "naïve — 日本語 😀\nline two\r\n\ttabbed \"quoted\" \\ back")
    assert roundtrip(message) == (message, None)


def test_assistant_message_defaults_when_optional_fields_are_missing() -> None:
    message, meta = message_from_obj({"type": "message", "role": "assistant", "content": "hi"})
    assert message == Message("assistant", "hi")
    assert meta is None


def test_unicode_is_written_unescaped() -> None:
    assert "日本語" in dumps(message_to_obj(Message("user", "日本語"), None))


def test_dumps_is_one_line_without_a_trailing_newline() -> None:
    line = dumps(message_to_obj(Message("user", "a\nb\r\nc\u2028d"), None))
    assert "\n" not in line and "\r" not in line
    assert not line.endswith("\n")
    assert json.loads(line)["content"] == "a\nb\r\nc\u2028d"


def test_dumps_uses_compact_separators() -> None:
    assert dumps({"type": "x", "a": 1}) == '{"type":"x","a":1}'


def test_message_object_starts_with_type_then_role() -> None:
    for message in (Message("user", "u"), Message("assistant", "a"), Message("tool", "t", tool_call_id="1", name="n")):
        assert dumps(message_to_obj(message, None)).startswith(f'{{"type":"message","role":"{message.role}","content"')


def test_message_object_has_the_documented_keys() -> None:
    obj = message_to_obj(Message("tool", "t", tool_call_id="c", name="read"), ToolMeta("read a", "1 lines"))
    assert list(obj) == ["type", "role", "content", "tool_calls", "tool_call_id", "name", "meta"]
    assert obj["meta"] == {"label": "read a", "summary": "1 lines"}
    assert obj["tool_calls"] == []


def test_header_object_starts_with_type_then_version() -> None:
    header = SessionHeader("01ABC", "/tmp/w", "2026-11-12T15:03:18Z")
    assert header.version == FORMAT_VERSION == 1
    assert dumps(header.to_obj()) == (
        '{"type":"session","version":1,"id":"01ABC","workspace":"/tmp/w","created_at":"2026-11-12T15:03:18Z"}'
    )


def test_header_round_trips() -> None:
    header = SessionHeader("01ABC", "/tmp/w", "2026-11-12T15:03:18Z")
    assert SessionHeader.from_obj(json.loads(dumps(header.to_obj()))) == header


@pytest.mark.parametrize(
    "obj",
    [
        {"type": "session", "id": "a", "workspace": "/w", "created_at": "x"},  # no version
        {"type": "session", "version": 1, "workspace": "/w", "created_at": "x"},  # no id
        {"type": "session", "version": "1", "id": "a", "workspace": "/w", "created_at": "x"},
        {"type": "message", "version": 1, "id": "a", "workspace": "/w", "created_at": "x"},
        {"type": "session", "version": 1, "id": "a", "workspace": 3, "created_at": "x"},
        {"type": "session", "version": True, "id": "a", "workspace": "/w", "created_at": "x"},
        [],
    ],
)
def test_header_from_obj_rejects_malformed_objects(obj: Any) -> None:
    with pytest.raises(RecordError):
        SessionHeader.from_obj(obj)


def test_header_from_obj_keeps_an_unknown_version_for_the_caller_to_judge() -> None:
    header = SessionHeader.from_obj(
        {"type": "session", "version": 2, "id": "a", "workspace": "/w", "created_at": "x"}
    )
    assert header.version == 2


@pytest.mark.parametrize(
    "obj",
    [
        {"type": "message", "content": "x"},  # no role
        {"type": "message", "role": "robot", "content": "x"},
        {"type": "message", "role": "system", "content": "x"},
        {"type": "title", "role": "user", "content": "x"},  # wrong type
        {"role": "user", "content": "x"},  # no type
        {"type": "message", "role": "user", "content": 5},
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": "nope"},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [1]},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "n"}]},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "n", "arguments": []}]},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [{"id": 1, "name": "n", "arguments": {}}]},
        {"type": "message", "role": "tool", "content": "x", "tool_call_id": 4},
        {"type": "message", "role": "tool", "content": "x", "name": 4},
        {"type": "message", "role": "tool", "content": "x", "meta": {"label": "a"}},
        {"type": "message", "role": "tool", "content": "x", "meta": "a"},
        "message",
        None,
    ],
)
def test_message_from_obj_rejects_malformed_objects(obj: Any) -> None:
    with pytest.raises(RecordError):
        message_from_obj(obj)


def test_record_error_is_a_value_error() -> None:
    assert issubclass(RecordError, ValueError)


def test_title_from_matches_the_ac3_example() -> None:
    title = title_from(AC3_TEXT)
    assert title == "Add a --verbose flag to health check and also upd…"
    assert title[:-1] == "Add a --verbose flag to health check and also upd"
    assert len(title) == 50  # the example's 49 characters plus the ellipsis


def test_title_from_keeps_a_text_of_exactly_50_characters() -> None:
    text = "x" * 50
    assert title_from(text) == text


def test_title_from_cuts_a_text_of_51_characters() -> None:
    assert title_from("x" * 51) == "x" * 49 + "…"


def test_a_cut_title_is_never_longer_than_50_characters() -> None:
    assert len(title_from("word " * 100)) == 50


def test_title_from_collapses_whitespace_in_short_text() -> None:
    assert title_from("  hi \n\t there ") == "hi there"


def test_title_from_blank_text_is_empty() -> None:
    assert title_from("  \n\t ") == ""


def test_dumps_escapes_a_lone_surrogate_so_the_line_is_valid_utf8() -> None:
    line = dumps({"type": "message", "role": "user", "content": "a\ud800b"})
    assert line == '{"type":"message","role":"user","content":"a\\ud800b"}'
    line.encode("utf-8")  # must not raise
    assert json.loads(line)["content"] == "a\ud800b"


def test_dumps_keeps_non_ascii_text_readable_when_there_is_no_surrogate() -> None:
    assert dumps({"type": "x", "a": "日本語"}) == '{"type":"x","a":"日本語"}'
