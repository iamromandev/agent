"""Questions the session asks its clients (approvals now; the trust prompts later): the first valid answer wins."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal

from core.engine.approval import ApprovalRequest, Decision
from core.engine.session.ulid import new_ulid
from core.host.bus import EventBus
from core.trust import TrustAsk
from core.wire import models as wm
from core.wire.codec import WireError, validate_answer

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


def _wire_ask(event: AskEvent) -> wm.Ask:
    request = event.request
    if isinstance(request, TrustAsk):
        body: wm.ApprovalRequest | wm.TrustAsk = wm.TrustAsk(kind=request.kind, lines=request.lines, note=request.note)
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
    return wm.Ask(v=1, ask_id=event.ask_id, kind=event.kind, request=body)


def _wire_answer(answer: Answer) -> wm.Answer:
    if isinstance(answer, bool):
        return wm.TrustAnswer(trust=answer)
    return wm.DecisionAnswer(choice=answer.choice, note=answer.note)


class AskBroker:
    def __init__(self, bus: EventBus) -> None:
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
            validate_answer(_wire_ask(event), _wire_answer(answer))
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
