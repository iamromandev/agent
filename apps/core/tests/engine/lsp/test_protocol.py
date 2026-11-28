"""The JSON-RPC framing: what is read is bounded and a bad frame is a ProtocolError."""

from __future__ import annotations

import asyncio

import pytest
from core.engine.lsp.protocol import MAX_MESSAGE, ProtocolError, encode, read_message


def reader(data: bytes) -> asyncio.StreamReader:
    stream = asyncio.StreamReader()
    stream.feed_data(data)
    stream.feed_eof()
    return stream


async def test_a_message_round_trips() -> None:
    assert await read_message(reader(encode({"id": 1, "method": "x"}))) == {"id": 1, "method": "x"}


@pytest.mark.parametrize(
    "data",
    [
        b"Content-Type: x\r\n\r\n{}",  # no Content-Length
        b"Content-Length: 4\r\n\r\nnope",  # invalid JSON
        b"Content-Length: 2\r\n\r\n[]",  # JSON but not an object
        f"Content-Length: {MAX_MESSAGE + 1}\r\n\r\n".encode(),  # too large
        b"X: " + b"a" * 9000 + b"\r\n\r\n",  # header over 8 KiB
        b"Content-Length: 50\r\n\r\n{}",  # cut short
    ],
)
async def test_a_bad_frame_raises_protocol_error(data: bytes) -> None:
    with pytest.raises(ProtocolError):
        await read_message(reader(data))
