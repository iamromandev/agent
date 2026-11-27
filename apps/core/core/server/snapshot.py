"""A session's state for a client that attaches (or fell too far behind): what the wire `Snapshot` carries."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

from core.engine.session.records import message_to_obj
from core.host.asks import AskEvent, ask_to_wire
from core.host.session import SessionHost
from core.wire import models as wm


def _redacted(value: object, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redacted(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item, redact) for item in value]
    return value


def wire_record(record: dict[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    """A session message record as the wire may carry it: image bytes dropped (format, name and size stay) and every
    text redacted (a tool result was redacted by the engine already; the user's and the model's text were not)."""
    images = record.get("images")
    out = {key: value for key, value in record.items() if key != "images"}
    out = _redacted(out, redact)
    if images:
        out["images"] = [
            {"format": i.get("format"), "name": redact(str(i.get("name", ""))), "size": i.get("size", 0)} for i in images
        ]
    return out


def records(host: SessionHost, redact: Callable[[str], str]) -> list[dict[str, Any]]:
    agent = host.agent
    out: list[dict[str, Any]] = []
    for message in agent.history:
        record = message_to_obj(message, agent.tool_meta.get(message.tool_call_id or ""))
        if message.images:  # metadata only: the bytes never leave the daemon
            record["images"] = [{"format": i.format, "name": i.name, "size": len(i.data)} for i in message.images]
        out.append(wire_record(record, redact))
    return out


def usage(host: SessionHost, redact: Callable[[str], str]) -> dict[str, Any]:
    totals = host.agent.usage
    return {
        "by_model": [{"model": redact(name), **dataclasses.asdict(model)} for name, model in totals.by_model],
        "unrecorded": host.agent.usage_unrecorded,
    }


def snapshot(
    host: SessionHost, *, seq: int, redact: Callable[[str], str], extra_asks: Sequence[AskEvent] = ()
) -> wm.Snapshot:
    """Everything a client needs to draw the session as it is at `seq`. Asks are passed on unredacted (see
    `ask_to_wire`); everything else that holds text is redacted."""
    agent = host.agent
    return wm.Snapshot(
        session_id=host.session_id,
        seq=seq,
        model=redact(host.model),
        mode=str(agent.mode),
        busy=host.turn_active,
        claims=wm.Claims(
            turn_active=host.turn_active,
            switching=host.switching,
            undoing=host.undoing,
            workflow_active=host.workflow_active,
            queueing=host.queueing,
        ),
        queue=[wm.QueueEntry(typed=redact(q.typed), label=redact(q.label or ""), client=q.client) for q in host.queue.entries],
        todos=[wm.TodoItem(content=redact(t.content), status=t.status) for t in agent.todos],
        usage=usage(host, redact),
        asks=[ask_to_wire(a, session_id=host.session_id) for a in (*host.asks.pending(), *extra_asks)],
        records=records(host, redact),
    )
