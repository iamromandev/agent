"""A socket file left by a killed daemon is replaced; one somebody listens on, or a file that is no socket, is not."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest
from core.server.files import bind_socket
from core.server.paths import socket_path


def test_a_stale_socket_is_replaced(data_home: Path, tmp_path: Path) -> None:
    path = socket_path(tmp_path)
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()  # the file stays, nobody listens: what a killed daemon leaves
    old = os.lstat(path).st_ino

    sock = bind_socket(path)
    try:
        assert os.lstat(path).st_ino != old and stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    finally:
        sock.close()


def test_a_socket_in_use_or_a_plain_file_is_left_alone(data_home: Path, tmp_path: Path) -> None:
    path = socket_path(tmp_path)
    live = bind_socket(path)
    try:
        with pytest.raises(OSError, match="in use"):
            bind_socket(path)
    finally:
        live.close()
    path.unlink()
    path.write_text("not a socket")
    with pytest.raises(OSError, match="not a socket"):
        bind_socket(path)
    assert path.read_text() == "not a socket"
