"""The peer check at connection time: only a process of our own user gets an HTTP connection."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest
from core.server.protocol import PeerCheckedH11
from uvicorn.config import Config
from uvicorn.server import ServerState


class Transport:
    """Enough of an asyncio transport for `connection_made`."""

    def __init__(self, sock: object) -> None:
        self.sock = sock
        self.closed = False

    def get_extra_info(self, name: str, default: object = None) -> object:
        return {"socket": self.sock, "peername": None, "sockname": None, "sslcontext": None}.get(name, default)

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed


async def _app(scope, receive, send) -> None:  # never reached: the tests stop at the connection
    raise AssertionError


@pytest.fixture
def pair() -> Iterator[socket.socket]:
    a, b = socket.socketpair(socket.AF_UNIX)
    with a, b:
        yield a


def _protocol(peer=None) -> tuple[PeerCheckedH11, ServerState]:
    state = ServerState()
    proto = PeerCheckedH11(config=Config(app=_app, http=PeerCheckedH11, log_level="error"), server_state=state, app_state={})
    if peer is not None:
        proto.peer_uid = staticmethod(peer)  # ty: ignore[invalid-assignment]
    return proto, state


async def test_our_own_uid_is_let_in(pair: socket.socket) -> None:
    proto, state = _protocol()
    transport = Transport(pair)
    proto.connection_made(transport)  # ty: ignore[invalid-argument-type]
    assert not transport.closed and proto in state.connections
    proto.connection_lost(None)
    assert proto not in state.connections


@pytest.mark.parametrize(
    "case",
    ["other uid", "credential error", "credential bug", "tcp socket", "no socket"],
)
async def test_anything_else_is_closed_and_never_joins_the_connections(pair: socket.socket, case: str) -> None:
    def other(_sock: socket.socket) -> int:
        return os.getuid() + 1

    def error(_sock: socket.socket) -> int:
        raise OSError("no credentials")

    def bug(_sock: socket.socket) -> int:
        raise ZeroDivisionError

    peer = {"other uid": other, "credential error": error, "credential bug": bug}.get(case)
    proto, state = _protocol(peer)
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock = tcp if case == "tcp socket" else None if case == "no socket" else pair
        transport = Transport(sock)
        proto.connection_made(transport)  # ty: ignore[invalid-argument-type]
        assert transport.closed and proto not in state.connections
        # whatever asyncio calls next must be harmless
        proto.data_received(b"GET / HTTP/1.1\r\n\r\n")
        proto.eof_received()
        proto.pause_writing()
        proto.resume_writing()
        proto.connection_lost(None)
    finally:
        tcp.close()
