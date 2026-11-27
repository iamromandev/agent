"""The host's bus items on the wire: engine and host events through the codec, asks and their resolutions through
`core.host.asks`. A server encodes every item it streams with `encode`; a remote client decodes with `decode`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.host.asks import AskEvent, AskResolved, ask_from_wire, ask_to_wire, resolved_from_wire, resolved_to_wire
from core.wire import models as wm
from core.wire.codec import WireError, WireEvent, from_wire, to_wire


def encode(item: object, *, session_id: str | None, seq: int, redact: Callable[[str], str]) -> dict[str, Any] | None:
    """The wire dict of one bus item, or None when it has no wire form (the workflow events, which only the local app
    draws) or could not be redacted (it is then never sent: there is no unredacted fallback)."""
    if isinstance(item, AskEvent):
        return ask_to_wire(item, session_id=session_id, seq=seq).model_dump(mode="json")
    if isinstance(item, AskResolved):
        return resolved_to_wire(item, session_id=session_id, seq=seq).model_dump(mode="json")
    try:
        return to_wire(item, session_id=session_id, seq=seq, redact=redact)  # ty: ignore[invalid-argument-type]
    except WireError:
        return None


def decode(data: dict[str, Any]) -> WireEvent | AskEvent | AskResolved | wm.Message:
    """An event as the local app gets it; the other messages (snapshot, overflow, ...) as wire models. `WireError` for
    anything that is not a valid version 1 message."""
    message = from_wire(data)
    if isinstance(message, wm.Ask):
        return ask_from_wire(message)
    if isinstance(message, wm.AskResolved):
        return resolved_from_wire(message)
    return message
