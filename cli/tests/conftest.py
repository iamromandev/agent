from __future__ import annotations

import builtins
import io
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from src.config import get_settings

FIXTURES = Path(__file__).parent / "fixtures"

# Resolved at import, before any test touches HOME or XDG_DATA_HOME.
REAL_DATA_DIRS = tuple(
    {
        (Path.home() / ".local" / "share").resolve(),
        *([Path(os.environ["XDG_DATA_HOME"]).resolve()] if os.environ.get("XDG_DATA_HOME") else []),
    }
)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_WRITE_MODE_CHARS = frozenset("wax+")


class RealDataDirWriteError(AssertionError):
    """A test tried to write under the real user data folder."""


def _is_real_data_path(path: Any) -> bool:
    if isinstance(path, int):
        return False
    try:
        resolved = Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError):
        return False
    return any(resolved == real or real in resolved.parents for real in REAL_DATA_DIRS)


class NetworkAccessError(AssertionError):
    """A test tried to use a real network transport."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any real HTTP call. Tests use httpx.MockTransport instead (NFR2)."""

    async def deny_async(self: Any, request: httpx.Request) -> httpx.Response:
        raise NetworkAccessError(f"real network call blocked: {request.url}")

    def deny_sync(self: Any, request: httpx.Request) -> httpx.Response:
        raise NetworkAccessError(f"real network call blocked: {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_sync)


@pytest.fixture(autouse=True)
def _sessions_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point sessions at a per-test folder and refuse writes under the real data folder (NFR5, AC19)."""
    base = tmp_path_factory.mktemp("sessions-env")
    monkeypatch.setenv("AGENT_SESSIONS_DIR", str(base / "sessions"))
    monkeypatch.setenv("XDG_DATA_HOME", str(base / "xdg"))
    get_settings.cache_clear()

    real_open, real_mkdir = os.open, os.mkdir
    real_symlink, real_rename, real_replace, real_chmod = os.symlink, os.rename, os.replace, os.chmod
    real_builtin_open = builtins.open

    def refuse(path: Any, what: str) -> None:
        if _is_real_data_path(path):
            raise RealDataDirWriteError(f"test tried to {what} {path}")

    def guarded_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & _WRITE_FLAGS:
            refuse(path, "write")
        return real_open(path, flags, *args, **kwargs)

    def guarded_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        refuse(path, "create")
        real_mkdir(path, *args, **kwargs)

    def guarded_symlink(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        refuse(dst, "link")
        real_symlink(src, dst, *args, **kwargs)

    def guarded_rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        refuse(src, "move")
        refuse(dst, "move to")
        real_rename(src, dst, *args, **kwargs)

    def guarded_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        refuse(src, "move")
        refuse(dst, "move to")
        real_replace(src, dst, *args, **kwargs)

    def guarded_chmod(path: Any, *args: Any, **kwargs: Any) -> None:
        refuse(path, "chmod")
        real_chmod(path, *args, **kwargs)

    def guarded_builtin_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if _WRITE_MODE_CHARS & set(mode):
            refuse(file, "write")
        return real_builtin_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    monkeypatch.setattr(os, "symlink", guarded_symlink)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "chmod", guarded_chmod)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_builtin_open)  # pathlib opens files through io.open
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    def resolve(name: str) -> Path:
        return FIXTURES / name

    return resolve
