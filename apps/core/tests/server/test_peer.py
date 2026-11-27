import os
import socket
import struct

import pytest
from core.server.peer import _parse_ucred, _parse_xucred, peer_uid


def test_peer_uid_of_a_socketpair_is_our_uid() -> None:
    a, b = socket.socketpair(socket.AF_UNIX)
    with a, b:
        assert peer_uid(a) == os.getuid()


def test_ucred_uid_is_unsigned() -> None:
    assert _parse_ucred(struct.pack("iII", 5, 2**31 + 7, 9)) == 2**31 + 7


def test_short_ucred_raises() -> None:
    with pytest.raises(OSError):
        _parse_ucred(b"\x00" * 4)


def test_xucred_reads_uid_and_rejects_a_bad_version() -> None:
    assert _parse_xucred(struct.pack("=II", 0, 501) + b"\x00" * 68) == 501
    with pytest.raises(OSError):
        _parse_xucred(struct.pack("=II", 1, 501) + b"\x00" * 68)
    with pytest.raises(OSError):
        _parse_xucred(b"\x00" * 4)


def test_a_non_unix_socket_is_refused() -> None:
    class Fake:
        family = socket.AF_INET

    with pytest.raises(OSError):
        peer_uid(Fake())  # ty: ignore[invalid-argument-type]
