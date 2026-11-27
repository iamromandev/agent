"""uvicorn's HTTP/1.1 protocol with a peer check: a connection from a process of another user, from anything that is
not a Unix socket, or whose credentials cannot be read, is closed before it reaches uvicorn (see the spike notes in
`core.server`)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket

from uvicorn.protocols.http.h11_impl import H11Protocol

from core.server.peer import peer_uid

log = logging.getLogger("agent.daemon")


class PeerCheckedH11(H11Protocol):
    peer_uid = staticmethod(peer_uid)  # tests swap it for a fake credential function
    _rejected = False

    def _allowed(self, transport: asyncio.BaseTransport) -> bool:
        sock = transport.get_extra_info("socket")
        if sock is None or sock.family != socket.AF_UNIX:
            return False
        return self.peer_uid(sock) == os.getuid()

    def connection_made(self, transport: asyncio.Transport) -> None:
        try:
            allowed = self._allowed(transport)
        except Exception:  # asyncio only logs an exception here and keeps the transport open: never let one out
            allowed = False
        if not allowed:
            # Never handed to the base: the connection stays out of uvicorn's connection set.
            self._rejected = True
            log.warning("refused a connection from another user or an unknown peer")
            with contextlib.suppress(Exception):
                transport.close()
            return
        super().connection_made(transport)

    def connection_lost(self, exc: Exception | None) -> None:
        if self._rejected:
            return  # the base would close a transport it never got
        super().connection_lost(exc)

    def data_received(self, data: bytes) -> None:
        if self._rejected:
            return
        super().data_received(data)

    def eof_received(self) -> None:
        if self._rejected:
            return
        super().eof_received()

    def pause_writing(self) -> None:
        if self._rejected:
            return
        super().pause_writing()

    def resume_writing(self) -> None:
        if self._rejected:
            return
        super().resume_writing()
