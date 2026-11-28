"""JSON-RPC framing for language servers (`Content-Length` headers over stdio), bounded in both header and body."""

from __future__ import annotations

import asyncio
import json

MAX_MESSAGE = 16 * 1024 * 1024
MAX_HEADER = 8 * 1024


class ProtocolError(Exception):
    """The server sent something that is not a valid frame."""


def encode(obj: dict[str, object]) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body


async def read_message(reader: asyncio.StreamReader) -> dict[str, object]:
    """The next message. Raises `ProtocolError` for a bad frame and `EOFError` when the stream ends cleanly."""
    length: int | None = None
    used = 0
    first = True
    while True:
        try:
            line = await reader.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise ProtocolError("header line too long") from exc
        if not line:
            if first:
                raise EOFError
            raise ProtocolError("stream ended inside a header")
        first = False
        used += len(line)
        if used > MAX_HEADER:
            raise ProtocolError("header over 8 KiB")
        text = line.strip()
        if not text:
            break
        name, _, value = text.partition(b":")
        if name.strip().lower() == b"content-length":
            try:
                length = int(value.strip())
            except ValueError as exc:
                raise ProtocolError("bad Content-Length") from exc
    if length is None or length < 0:
        raise ProtocolError("no Content-Length")
    if length > MAX_MESSAGE:
        raise ProtocolError(f"message over {MAX_MESSAGE} bytes")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise ProtocolError("stream ended inside a message") from exc
    try:
        value = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("message is not an object")
    return value
