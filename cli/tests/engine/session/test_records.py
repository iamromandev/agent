from __future__ import annotations

import json
from typing import Any

import pytest
from src.engine.provider.base import Message
from src.engine.session.records import (
    RecordError,
    SessionHeader,
    ToolMeta,
    dumps,
    message_from_obj,
    message_to_obj,
)

AC3_TEXT = "  Add   a --verbose\nflag to health check and also update the docs please  "


def roundtrip(message: Message, meta: ToolMeta | None = None) -> tuple[Message, ToolMeta | None]:
    obj = json.loads(dumps(message_to_obj(message, meta)))
    return message_from_obj(obj)


@pytest.mark.parametrize(
    "obj",
    [
        {"type": "session", "version": 1, "id": "a", "workspace": 3, "created_at": "x"},
        {"type": "session", "version": True, "id": "a", "workspace": "/w", "created_at": "x"},
        [],
    ],
)
def test_header_from_obj_rejects_malformed_objects(obj: Any) -> None:
    with pytest.raises(RecordError):
        SessionHeader.from_obj(obj)


@pytest.mark.parametrize(
    "obj",
    [
        {"role": "user", "content": "x"},  # no type
        {"type": "message", "role": "user"},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": "nope"},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [1]},
        {"type": "message", "role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "n"}]},
        {"type": "message", "role": "tool", "content": "x", "tool_call_id": 4},
        {"type": "message", "role": "tool", "content": "x", "name": 4},
        {"type": "message", "role": "tool", "content": "x", "meta": "a"},
        None,
    ],
)
def test_message_from_obj_rejects_malformed_objects(obj: Any) -> None:
    with pytest.raises(RecordError):
        message_from_obj(obj)


def test_dumps_escapes_a_lone_surrogate_so_the_line_is_valid_utf8() -> None:
    line = dumps({"type": "message", "role": "user", "content": "a\ud800b"})
    assert line == '{"type":"message","role":"user","content":"a\\ud800b"}'
    line.encode("utf-8")  # must not raise
    assert json.loads(line)["content"] == "a\ud800b"
