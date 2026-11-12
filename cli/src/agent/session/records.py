"""Session file records: one JSON object per line, with a fixed key order.

`type` is always the first key, and `role` the second for messages, so listing can
count lines by substring instead of parsing them (NFR2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, get_args

from src.agent.provider.base import Message, Role, ToolCall

FORMAT_VERSION = 1
TITLE_LIMIT = 50
_MESSAGE_ROLES = frozenset(get_args(Role)) - {"system"}


class RecordError(ValueError):
    """A record that does not match the session format."""


@dataclass(frozen=True)
class ToolMeta:
    """How a tool message is shown and pruned."""

    label: str  # "read src/main.py"
    summary: str  # "31 lines"


@dataclass(frozen=True)
class SessionHeader:
    id: str
    workspace: str
    created_at: str  # ISO-8601 UTC, "2026-11-12T15:03:18Z"
    version: int = FORMAT_VERSION

    def to_obj(self) -> dict[str, Any]:
        return {
            "type": "session",
            "version": self.version,
            "id": self.id,
            "workspace": self.workspace,
            "created_at": self.created_at,
        }

    @classmethod
    def from_obj(cls, obj: Any) -> SessionHeader:
        """Parse a header; an unknown version is kept so the caller can refuse it."""
        if not isinstance(obj, dict) or obj.get("type") != "session":
            raise RecordError("not a session header")
        version = obj.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise RecordError("header has no integer version")
        fields = {key: obj.get(key) for key in ("id", "workspace", "created_at")}
        if not all(isinstance(value, str) for value in fields.values()):
            raise RecordError("header is missing id, workspace or created_at")
        return cls(version=version, **fields)


def message_to_obj(m: Message, meta: ToolMeta | None, *, embedded: bool = False) -> dict[str, Any]:
    """The record for a message. `embedded` drops `type`, for use inside a compaction record."""
    obj: dict[str, Any] = {} if embedded else {"type": "message"}
    obj.update(
        role=m.role,
        content=m.content,
        tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls],
        tool_call_id=m.tool_call_id,
        name=m.name,
        meta={"label": meta.label, "summary": meta.summary} if meta else None,
    )
    return obj


def message_from_obj(obj: Any, *, embedded: bool = False) -> tuple[Message, ToolMeta | None]:
    if not isinstance(obj, dict):
        raise RecordError("message record is not an object")
    if not embedded and obj.get("type") != "message":
        raise RecordError("not a message record")
    role = obj.get("role")
    if role not in _MESSAGE_ROLES:
        raise RecordError(f"bad role: {role!r}")
    content = obj.get("content")
    if not isinstance(content, str):
        raise RecordError("message content is not a string")
    tool_call_id, name = obj.get("tool_call_id"), obj.get("name")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise RecordError("tool_call_id is not a string")
    if name is not None and not isinstance(name, str):
        raise RecordError("name is not a string")
    message = Message(role, content, _tool_calls(obj.get("tool_calls")), tool_call_id, name)
    return message, _meta(obj.get("meta"))


def _tool_calls(raw: Any) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RecordError("tool_calls is not a list")
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RecordError("tool call is not an object")
        call_id, call_name, arguments = item.get("id"), item.get("name"), item.get("arguments")
        if not isinstance(call_id, str) or not isinstance(call_name, str) or not isinstance(arguments, dict):
            raise RecordError("tool call needs a string id, a string name and object arguments")
        calls.append(ToolCall(call_id, call_name, arguments))
    return calls


def _meta(raw: Any) -> ToolMeta | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("label"), str) or not isinstance(raw.get("summary"), str):
        raise RecordError("meta needs a string label and summary")
    return ToolMeta(raw["label"], raw["summary"])


def dumps(obj: dict[str, Any]) -> str:
    """One line, no trailing newline. `json` escapes control characters, so it can't contain a raw newline.

    A lone surrogate (from a tool reading a non-UTF-8 file, say) can't be written as UTF-8, so that record
    falls back to ASCII-only escapes: `\\ud800` reads back as the exact original string. Key order is kept.
    """
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return text


def title_from(text: str) -> str:
    """The first user message with whitespace collapsed; over 50 characters it becomes 49 plus "…" (R5, AC3).

    The spec's AC3 example ends "...also upd…", which is 49 characters plus the ellipsis, 50 in total.
    """
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= TITLE_LIMIT else collapsed[: TITLE_LIMIT - 1] + "…"


def compaction_notice(pruned: int, summarized: int, before: int | None = None, after: int | None = None) -> str:
    """The transcript line for a compaction (R18). The live notice and the replayed one are both made here."""
    text = f"Context compacted: pruned {pruned} tool output{'' if pruned == 1 else 's'}"
    if summarized:
        text += f", summarized {summarized} message{'' if summarized == 1 else 's'}"
    if before is not None and after is not None:
        text += f" (ctx ~{before} \u2192 ~{after})"
    return text
