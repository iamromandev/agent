"""Listing and loading saved sessions for one workspace.

Listing reads each file once and counts message lines by substring instead of parsing
them (NFR2). That is exact because records are written with a fixed key order and JSON
escapes newlines, so `\\n{"type":"message","role":"user"` can only be the start of a record.
"""

from __future__ import annotations

import builtins
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.agent.provider.base import Message
from src.agent.session.paths import workspace_dir
from src.agent.session.records import (
    FORMAT_VERSION,
    RecordError,
    SessionHeader,
    ToolMeta,
    compaction_notice,
    message_from_obj,
)

MIN_PREFIX = 6
UNTITLED = "(untitled)"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
_TITLE_MARK = b'\n{"type":"title"'
_COUNTED_MARKS = (b'\n{"type":"message","role":"user"', b'\n{"type":"message","role":"assistant"')


@dataclass(frozen=True)
class SessionInfo:
    id: str
    path: Path
    title: str  # "(untitled)" if no title record
    created_at: datetime
    updated_at: datetime  # file mtime (about the last record), timezone-aware
    message_count: int  # user + assistant message records


@dataclass(frozen=True)
class TranscriptItem:
    kind: Literal["user", "assistant", "tool", "notice"]
    text: str  # user text / assistant markdown / tool label / notice text
    summary: str = ""  # tool result summary


@dataclass(frozen=True)
class LoadedSession:
    info: SessionInfo
    history: list[Message]
    tool_meta: dict[str, ToolMeta]  # by tool_call_id
    items: list[TranscriptItem]
    skipped_lines: int  # complete-but-invalid lines skipped (a cut last line is not counted)


class SessionError(Exception):
    """A user-facing message about a session that can't be used."""


class SessionNotFound(SessionError): ...


class AmbiguousSession(SessionError): ...


class SessionStore:
    def __init__(self, root: Path, workspace: Path) -> None:
        self._workspace = workspace.resolve()
        self._directory = workspace_dir(root, workspace)
        self._skipped: list[Path] = []

    @property
    def skipped(self) -> builtins.list[Path]:
        """Files the last `list()` left out because they could not be used."""
        return list(self._skipped)

    def list(self) -> builtins.list[SessionInfo]:  # `list` is shadowed by this method in the class body
        """This workspace's sessions, most recently updated first."""
        infos: list[SessionInfo] = []
        skipped: list[Path] = []
        try:
            # Real folders are not sessions and are ignored. Symlinks and special files are reported by _read.
            paths = sorted(p for p in self._directory.glob("*.jsonl") if p.is_symlink() or not p.is_dir())
        except OSError:
            paths = []
        for path in paths:
            try:
                header, data, mtime = self._read(path)
                infos.append(self._info(path, header, data, mtime))
            except (SessionError, OSError):
                skipped.append(path)
        self._skipped = skipped
        return sorted(infos, key=lambda i: (i.updated_at, i.id), reverse=True)

    def latest(self) -> SessionInfo | None:
        infos = self.list()
        return infos[0] if infos else None

    def find(self, prefix: str) -> SessionInfo:
        if len(prefix) < MIN_PREFIX:
            raise SessionNotFound(f'Session id "{prefix}" is too short: give at least {MIN_PREFIX} characters')
        wanted = prefix.upper()
        matches = [info for info in self.list() if info.id.upper().startswith(wanted)]
        if not matches:
            raise SessionNotFound(f'No saved session "{prefix}" for this folder')
        if len(matches) > 1:
            raise AmbiguousSession(f'Session id "{prefix}" matches {len(matches)} sessions: give more characters')
        return matches[0]

    def load(self, session_id: str) -> LoadedSession:
        if not _SAFE_ID.fullmatch(session_id):
            raise SessionNotFound(f'No saved session "{session_id}" for this folder')
        path = self._directory / f"{session_id}.jsonl"
        try:
            header, data, mtime = self._read(path)
        except FileNotFoundError as error:
            raise SessionNotFound(f'No saved session "{session_id}" for this folder') from error
        except OSError as error:
            raise SessionError(f"Could not read session {session_id}: {error.strerror or error}") from error
        info = self._info(path, header, data, mtime)
        return self._replay(info, data)

    def _read(self, path: Path) -> tuple[SessionHeader, bytes, float]:
        """Read a file and check its header. Raises SessionError when it can't be used.

        The file is opened without following a symlink, and must be a regular file: a FIFO would block
        the read, and a link could point outside the sessions folder.
        """
        name = path.stem
        if not _SAFE_ID.fullmatch(name):
            raise SessionError(f"Session file {path.name} does not have a valid session id as its name")
        data, mtime = _read_regular_file(path)
        end = data.find(b"\n")
        if end < 0:
            raise SessionError(f"Session {name} has an unreadable header")
        try:
            header = SessionHeader.from_obj(json.loads(data[:end]))
            _parse_time(header.created_at)
        except (ValueError, RecursionError) as error:  # json errors, bad UTF-8, RecordError, a bad date, deep nesting
            raise SessionError(f"Session {name} has an unreadable header") from error
        if header.version != FORMAT_VERSION:
            raise SessionError(
                f"Session {name} uses format version {header.version}, which this agent does not understand"
            )
        if header.id != name:  # a copied or renamed file: the file name is the id that load() can find
            raise SessionError(f"Session {name} was copied or renamed: its header says {header.id}")
        if header.workspace != str(self._workspace):
            raise SessionError(f"Session {name} belongs to a different workspace")
        return header, data, mtime

    def _info(self, path: Path, header: SessionHeader, data: bytes, mtime: float) -> SessionInfo:
        complete = data[: data.rfind(b"\n") + 1]  # a cut last line is not a record yet
        created_at = _parse_time(header.created_at)
        return SessionInfo(
            id=path.stem,
            path=path,
            title=_last_title(complete),
            created_at=created_at,
            updated_at=_updated_at(mtime, created_at),
            message_count=sum(complete.count(mark) for mark in _COUNTED_MARKS),
        )

    def _replay(self, info: SessionInfo, data: bytes) -> LoadedSession:
        complete = data[: data.rfind(b"\n")]  # drops a cut last line, if there is one
        history: list[Message] = []
        tool_meta: dict[str, ToolMeta] = {}
        items: list[TranscriptItem] = []
        skipped = 0
        for line in complete.split(b"\n")[1:]:  # the first line is the header
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                kind = obj.get("type") if isinstance(obj, dict) else None
                if kind == "message":
                    message, meta = message_from_obj(obj)
                    history.append(message)
                    _remember(tool_meta, message, meta)
                    if item := _item(message, meta):
                        items.append(item)
                elif kind == "compaction":
                    history, notice = _compaction(obj, tool_meta)
                    items.append(notice)
                elif kind is None:
                    raise RecordError("record without a type")
            except (ValueError, TypeError, AttributeError, RecursionError):  # bad JSON or UTF-8, RecordError
                skipped += 1
        return LoadedSession(info, history, tool_meta, items, skipped)


def _read_regular_file(path: Path) -> tuple[bytes, float]:
    """The bytes and mtime of a regular file, opened with `O_NOFOLLOW` (and `O_NONBLOCK`, so a FIFO can't hang us)."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SessionError(f"Session file {path.name} is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as file:
            return file.read(), info.st_mtime
    finally:
        os.close(fd)


def _parse_time(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _updated_at(mtime: float, fallback: datetime) -> datetime:
    """The file's mtime as a date, or `fallback` when the mtime is not one (a clock or filesystem gone wrong)."""
    try:
        return datetime.fromtimestamp(mtime, UTC)
    except (OverflowError, ValueError, OSError):
        return fallback


def _last_title(complete: bytes) -> str:
    end = len(complete)
    while (start := complete.rfind(_TITLE_MARK, 0, end)) >= 0:
        line_end = complete.find(b"\n", start + 1)
        try:
            title = json.loads(complete[start + 1 : line_end])["title"]
            if isinstance(title, str):
                return title
        except (ValueError, KeyError, TypeError, RecursionError):
            pass
        end = start
    return UNTITLED


def _remember(tool_meta: dict[str, ToolMeta], message: Message, meta: ToolMeta | None) -> None:
    if meta and message.tool_call_id:
        tool_meta[message.tool_call_id] = meta


def _item(message: Message, meta: ToolMeta | None) -> TranscriptItem | None:
    """The transcript entry for a message. An assistant turn that only calls tools has none."""
    if message.role == "tool":
        return TranscriptItem("tool", meta.label if meta else message.name or "", meta.summary if meta else "")
    if message.role == "assistant" and not message.content:
        return None
    return TranscriptItem("user" if message.role == "user" else "assistant", message.content)


def _size(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _compaction(obj: dict[str, Any], tool_meta: dict[str, ToolMeta]) -> tuple[list[Message], TranscriptItem]:
    raw, pruned, summarized = obj.get("history"), _size(obj.get("pruned")), _size(obj.get("summarized"))
    if not isinstance(raw, list) or pruned is None or summarized is None:
        raise RecordError("bad compaction record")
    parsed = [message_from_obj(entry, embedded=True) for entry in raw]
    for message, meta in parsed:
        _remember(tool_meta, message, meta)
    notice = compaction_notice(pruned, summarized, _size(obj.get("before")), _size(obj.get("after")))
    return [message for message, _ in parsed], TranscriptItem("notice", notice)
