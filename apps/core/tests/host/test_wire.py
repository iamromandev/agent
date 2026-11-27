"""What the host's bus items look like on the wire, and back."""

from __future__ import annotations

import json

from core.engine.approval import ApprovalRequest, Choice, Decision
from core.engine.events import Notice, WorkflowLine
from core.engine.tools.base import Preview
from core.host.asks import AskEvent, AskResolved
from core.host.wire import decode, encode
from core.trust import TrustAsk


def mask(text: str) -> str:
    return text.replace("hunter2hunter2", "[redacted:env:X]")


def _round(item: object) -> object:
    data = encode(item, session_id="s", seq=4, redact=mask)
    assert data is not None and data["seq"] == 4
    return decode(json.loads(json.dumps(data)))


def test_an_approval_ask_round_trips_and_its_preview_is_not_redacted_again() -> None:
    preview = Preview("command", "bash", "echo hunter2hunter2")  # the model's own command: shown as it will run
    ask = AskEvent("a1", "approval", ApprovalRequest("bash", "bash echo", preview, True, save_rule="bash(echo)"))
    assert _round(ask) == ask


def test_a_trust_ask_and_the_resolutions_round_trip() -> None:
    trust = AskEvent("a2", "mcp_trust", TrustAsk("mcp_trust", ["x  y"], header=["h"], decline="no", accept="yes"))
    assert _round(trust) == trust
    assert _round(AskResolved("a1", "B", Decision(Choice.ONCE, ""))) == AskResolved("a1", "B", Decision(Choice.ONCE, ""))
    assert _round(AskResolved("a2", "", None)) == AskResolved("a2", "", None)


def test_an_engine_event_is_redacted_and_one_with_no_wire_form_is_none() -> None:
    assert _round(Notice("info", "hunter2hunter2")) == Notice("info", "[redacted:env:X]")
    assert encode(WorkflowLine("info", "x"), session_id=None, seq=1, redact=mask) is None
    assert encode(object(), session_id=None, seq=1, redact=mask) is None
