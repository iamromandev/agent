"""The user id of the process at the other end of a Unix socket."""

import socket
import struct
import sys

SOL_LOCAL = 0  # macOS: level for LOCAL_PEERCRED
LOCAL_PEERCRED = 1
XUCRED_SIZE = 76  # sizeof(struct xucred)
XUCRED_VERSION = 0


def _parse_ucred(data: bytes) -> int:
    """The uid in a Linux `struct ucred` (pid is signed, uid and gid unsigned)."""
    fmt = "iII"
    if len(data) < struct.calcsize(fmt):
        raise OSError("short ucred")
    _pid, uid, _gid = struct.unpack_from(fmt, data)
    return uid


def _parse_xucred(data: bytes) -> int:
    """The uid in a macOS `struct xucred` (cr_version, cr_uid, ...)."""
    if len(data) < 8:
        raise OSError("short xucred")
    version, uid = struct.unpack_from("=II", data)
    if version != XUCRED_VERSION:
        raise OSError(f"unknown xucred version {version}")
    return uid


def peer_uid(sock: socket.socket) -> int:
    """The effective uid of the peer of a connected AF_UNIX socket; `OSError` when it cannot be told."""
    if sock.family != socket.AF_UNIX:
        raise OSError("not a Unix socket")
    if sys.platform == "linux":
        return _parse_ucred(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII")))
    if sys.platform == "darwin":
        return _parse_xucred(sock.getsockopt(SOL_LOCAL, LOCAL_PEERCRED, XUCRED_SIZE))
    raise OSError(f"peer credentials are not supported on {sys.platform}")
