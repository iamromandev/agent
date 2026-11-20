from __future__ import annotations

import base64
import json
import math
import os
from typing import Any

import pytest
from src.engine.provider.base import Image, Message
from src.engine.provider.images import MAX_IMAGE_BYTES
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


# --- images -----------------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"data"


def test_a_message_without_images_has_no_images_key() -> None:
    obj = message_to_obj(Message("user", "hi"), None)
    assert list(obj) == ["type", "role", "content", "tool_calls", "tool_call_id", "name", "meta"]


def test_images_come_after_meta_and_before_at() -> None:
    obj = message_to_obj(Message("user", "hi", images=(Image("png", PNG, "a.png"),)), None)
    assert list(obj)[:2] == ["type", "role"]
    assert list(obj)[-2:] == ["meta", "images"]
    line = dumps({**obj, "at": "x"})
    assert line.index('"images"') < line.index('"at"')
    assert obj["images"] == [{"format": "png", "name": "a.png", "data": base64.b64encode(PNG).decode()}]


def test_images_round_trip() -> None:
    images = (Image("png", PNG, "a.png"), Image("jpeg", b"\xff\xd8\xffx", "clipboard"))
    message, _ = roundtrip(Message("tool", "image a.png", tool_call_id="c", name="read", images=images))
    assert message.images == images


def test_a_record_without_images_loads_with_none() -> None:
    message, _ = message_from_obj({"type": "message", "role": "user", "content": "x"})
    assert message.images == ()


def _record(images: Any) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": "x", "images": images}


@pytest.mark.parametrize(
    "images",
    [
        "nope",
        [1],
        [{"format": "bmp", "name": "a", "data": "AAAA"}],
        [{"format": "png", "name": "a", "data": "!!!not base64!!!"}],
        [{"format": "png", "name": "a"}],
        [{"format": "png", "name": 3, "data": "AAAA"}],
        [{"format": "png", "name": "a", "data": base64.b64encode(b"\0" * (MAX_IMAGE_BYTES + 1)).decode()}],
    ],
)
def test_malformed_images_make_the_record_invalid(images: Any) -> None:
    with pytest.raises(RecordError):
        message_from_obj(_record(images))


def test_a_stored_image_stays_within_the_size_bound() -> None:
    data = PNG + os.urandom(100_000)
    line = dumps({**message_to_obj(Message("user", "", images=(Image("png", data, "a.png"),)), None), "at": STAMP})
    without = dumps({**message_to_obj(Message("user", ""), None), "at": STAMP})
    assert len(line) - len(without) <= math.ceil(4 / 3 * len(data)) + 200


STAMP = "2026-11-12T15:03:18Z"
