"""Questions the session asks its clients (approvals now; the trust prompts later): the first valid answer wins."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

from core.engine.approval import ApprovalRequest, Decision
from core.engine.session.ulid import new_ulid
from core.engine.tools.base import Preview
from core.host.bus import EventBus
from core.trust import TrustAsk
from core.wire import models as wm
from core.wire.codec import WireError, trust_from_wire, trust_to_wire, validate_answer

AskKind = Literal["approval", "protected", "global_trust", "mcp_trust", "permission_trust"]
Answer = Decision | bool  # a decision for an approval, trust yes/no for a trust prompt
AnswerResult = Literal["ok", "already_answered", "unknown", "invalid"]
REMEMBERED = 256  # answered asks kept to tell a late second answer from an unknown id


@dataclass(frozen=True)
class AskEvent:
    """A question for the clients (published on the bus; also listed by `AskBroker.pending`)."""

    ask_id: str
    kind: AskKind
    request: ApprovalRequest | TrustAsk


@dataclass(frozen=True)
class AskResolved:
    """An ask is over: answered by client `by`, or withdrawn (`by` empty, `decision` None)."""

    ask_id: str
    by: str
    decision: Answer | None


def ask_to_wire(event: AskEvent, *, session_id: str | None = None, seq: int = 0) -> wm.Ask:
    """The wire form of an ask. Nothing is redacted here, on purpose: a `diff` preview was already redacted by the
    engine (`Guardrails.redact_preview`, for the dialog), and a `command` or `content` preview is the model's own text
    the user must see as it will run or be written; redacting it again would show the user something other than what
    they approve."""
    request = event.request
    if isinstance(request, TrustAsk):
        body: wm.ApprovalRequest | wm.TrustAsk = trust_to_wire(request)
    else:
        preview = request.preview
        body = wm.ApprovalRequest(
            tool=request.tool,
            label=request.label,
            preview=None if preview is None else wm.PreviewModel(kind=preview.kind, title=preview.title, body=preview.body),
            allow_always=request.allow_always,
            save_rule=request.save_rule,
            protected=request.protected,
            protected_path=request.protected_path,
            subagent=request.subagent,
        )
    return wm.Ask(v=1, session_id=session_id, seq=seq, ask_id=event.ask_id, kind=event.kind, request=body)


def ask_from_wire(message: wm.Ask) -> AskEvent:
    """The ask a remote client shows: the same `ApprovalRequest` or `TrustAsk` the local dialogs take."""
    request = message.request
    if isinstance(request, wm.TrustAsk):
        return AskEvent(message.ask_id, message.kind, trust_from_wire(request))
    preview = request.preview
    return AskEvent(
        message.ask_id,
        message.kind,
        ApprovalRequest(
            request.tool,
            request.label,
            None if preview is None else Preview(preview.kind, preview.title, preview.body),
            request.allow_always,
            save_rule=request.save_rule,
            protected=request.protected,
            protected_path=request.protected_path,
            subagent=request.subagent,
        ),
    )


def answer_to_wire(answer: Answer) -> wm.Answer:
    if isinstance(answer, bool):
        return wm.TrustAnswer(trust=answer)
    return wm.DecisionAnswer(choice=answer.choice, note=answer.note)


def answer_from_wire(answer: wm.Answer) -> Answer:
    if isinstance(answer, wm.TrustAnswer):
        return answer.trust
    return Decision(answer.choice, answer.note)


def resolved_to_wire(event: AskResolved, *, session_id: str | None = None, seq: int = 0) -> wm.AskResolved:
    decision = None if event.decision is None else answer_to_wire(event.decision)
    return wm.AskResolved(session_id=session_id, seq=seq, ask_id=event.ask_id, by=event.by, decision=decision)


def resolved_from_wire(message: wm.AskResolved) -> AskResolved:
    decision = None if message.decision is None else answer_from_wire(message.decision)
    return AskResolved(message.ask_id, message.by, decision)


class Publisher(Protocol):
    """Where an ask goes: a session's bus, or the daemon's fan-out to every session (its trust prompts)."""

    def publish(self, event: object) -> object: ...


class AskBroker:
    def __init__(self, bus: EventBus | Publisher) -> None:
        self._bus = bus
        self._pending: dict[str, tuple[AskEvent, asyncio.Future[Answer]]] = {}
        self._answered: deque[str] = deque(maxlen=REMEMBERED)

    def _publish(self, event: object) -> None:
        self._bus.publish(event)

    async def ask(self, kind: AskKind, request: ApprovalRequest | TrustAsk) -> Answer:
        """Publish the question and wait for the first valid answer. Cancelled: the ask is withdrawn."""
        event = AskEvent(new_ulid(), kind, request)
        future: asyncio.Future[Answer] = asyncio.get_running_loop().create_future()
        self._pending[event.ask_id] = (event, future)
        self._publish(event)
        try:
            return await future
        finally:
            if self._pending.pop(event.ask_id, None) is not None:  # not answered: withdrawn
                self._publish(AskResolved(event.ask_id, "", None))

    def answer(self, ask_id: str, answer: Answer, client_id: str) -> AnswerResult:
        entry = self._pending.get(ask_id)
        if entry is None:
            return "already_answered" if ask_id in self._answered else "unknown"
        event, future = entry
        if future.done():  # its asker was cancelled and has not run since
            return "unknown"
        try:
            validate_answer(ask_to_wire(event), answer_to_wire(answer))  # only an answer the ask offers counts
        except WireError:
            return "invalid"
        del self._pending[ask_id]
        self._answered.append(ask_id)
        future.set_result(answer)
        self._publish(AskResolved(ask_id, client_id, answer))
        return "ok"

    def pending(self) -> list[AskEvent]:
        return [event for event, future in self._pending.values() if not future.done()]

    def cancel_all(self, reason: str) -> None:
        """Withdraw every open ask (an interrupt, the session closing): each asker gets CancelledError."""
        for ask_id, (_, future) in list(self._pending.items()):
            del self._pending[ask_id]
            future.cancel(reason)
            self._publish(AskResolved(ask_id, "", None))


class HostApprover:
    """The engine's `Approver` on top of the broker: every approval becomes an ask for the clients."""

    def __init__(self, broker: AskBroker) -> None:
        self.broker = broker

    async def ask(self, request: ApprovalRequest) -> Decision:
        answer = await self.broker.ask("protected" if request.protected else "approval", request)
        assert isinstance(answer, Decision)
        return answer
