"""Engine events to wire dicts and back."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, get_args

from pydantic import TypeAdapter, ValidationError

from core.engine import events as ev
from core.engine.approval import Choice
from core.engine.mode import Mode
from core.engine.todo import TodoItem
from core.engine.tools.base import Preview, ToolResult
from core.host import events as hev
from core.host.queue import Queued
from core.trust import TrustAsk
from core.wire import models as wm

# What crosses the wire as an event: the engine's events and the host's turn steps.
WireEvent = (
    ev.AgentEvent
    | hev.ProviderState
    | hev.ModelChanged
    | hev.QueueChanged
    | hev.ModeChanged
    | hev.SessionStarting
    | hev.TurnUndone
    | hev.UserEntry
    | hev.Delivering
    | hev.JobEventsDone
    | hev.EntryEnd
    | hev.TurnOutcome
)

_ADAPTER: TypeAdapter[Any] = TypeAdapter(wm.WireMessage)
_KNOWN: frozenset[str] = frozenset(
    arm.model_fields["type"].default for arm in get_args(get_args(wm.WireMessage)[0])
)


class WireError(Exception):
    """A message that is not version 1 or has an unknown type or shape."""


def _redact_value(value: object, redact: Callable[[str], str]) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {(redact(k) if isinstance(k, str) else k): _redact_value(v, redact) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v, redact) for v in value]
    return value


def _preview(preview: Preview | None, redact: Callable[[str], str]) -> wm.PreviewModel | None:
    if preview is None:
        return None
    return wm.PreviewModel(kind=preview.kind, title=redact(preview.title), body=redact(preview.body))


def _build(event: WireEvent, head: dict[str, Any], redact: Callable[[str], str]) -> wm.Message:
    match event:
        case hev.ProviderState():
            return wm.ProviderState(**head, problem=None if event.problem is None else redact(event.problem))
        case hev.ModelChanged():
            return wm.ModelChanged(**head, model=redact(event.model), window=event.window)
        case hev.QueueChanged():
            entries = [wm.QueueEntry(typed=redact(q.typed), label=redact(q.label or ""), client=q.client) for q in event.entries]
            return wm.QueueChanged(**head, entries=entries)
        case hev.ModeChanged():
            return wm.ModeChanged(**head, mode=event.mode.value)
        case hev.SessionStarting():
            return wm.SessionStarting(**head, active=event.active)
        case hev.TurnUndone():
            return wm.TurnUndone(**head, client=event.client)
        case hev.UserEntry():
            label = None if event.label is None else redact(event.label)
            return wm.UserEntry(**head, text=redact(event.text), label=label, client=event.client)
        case hev.Delivering():
            return wm.Delivering(**head)
        case hev.JobEventsDone():
            return wm.JobEventsDone(**head)
        case hev.EntryEnd():
            return wm.EntryEnd(**head, discard=event.discard)
        case hev.TurnOutcome():
            check = event.init_check
            return wm.TurnOutcome(
                **head,
                kind=event.kind,
                status=event.status,
                continued=event.continued,
                # the user's own draft, given back exactly (see the model)
                restore_text=event.restore_text,
                queue_restore=event.queue_restore,
                delivered=event.delivered,
                turn=event.turn,
                init_check=None if check is None else wm.InitCheck(level=check[0], text=redact(check[1])),
                ack=event.ack,
                client=event.client,
                interrupted_by=event.interrupted_by,
                restores=[wm.Restore(client=client, text=text) for client, text in event.restores],
            )
        case ev.AssistantText():
            # Each delta on its own: a secret the model writes in one piece is caught, one split across two deltas is
            # not (the same limit as a secret split across two tool results).
            return wm.AssistantText(**head, delta=redact(event.delta))
        case ev.ToolStarted():
            args = _redact_value(dict(event.args), redact)
            assert isinstance(args, dict)
            return wm.ToolStarted(
                **head, call_id=event.call_id, label=redact(event.label), task=event.task, name=event.name, args=args
            )
        case ev.ToolFinished():
            result = event.result
            return wm.ToolFinished(
                **head,
                call_id=event.call_id,
                output=result.output,
                display=result.display,
                summary=redact(result.summary),
                is_error=result.is_error,
                images=[wm.ImageMeta(format=i.format, name=i.name, size=len(i.data)) for i in result.images],
                preview=_preview(event.preview, redact),
                applied=event.applied,
            )
        case ev.Notice():
            return wm.Notice(**head, level=event.level, text=redact(event.text))
        case ev.UsageUpdate():
            return wm.UsageUpdate(
                **head,
                used=event.used,
                window=event.window,
                estimated=event.estimated,
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
            )
        case ev.Compacting():
            return wm.Compacting(**head, active=event.active)
        case ev.TurnEnded():
            return wm.TurnEnded(**head, status=event.status)
        case ev.TodosChanged():
            items = [wm.TodoItem(content=redact(i.content), status=i.status) for i in event.items]
            return wm.TodosChanged(**head, items=items)
        case ev.TaskProgress():
            inner = _build(event.event, head, redact)
            return wm.TaskProgress(**head, call_id=event.call_id, event=inner)  # ty: ignore[invalid-argument-type]
        case ev.ToolProgress():
            return wm.ToolProgress(**head, text=redact(event.text))
        case ev.Phase():
            return wm.Phase(
                **head, kind=event.kind, step=event.step, max_steps=event.max_steps, detail=redact(event.detail)
            )
        case _:
            raise WireError(f"{type(event).__name__} is not sent on the wire")


def to_wire(event: WireEvent, *, session_id: str | None, seq: int, redact: Callable[[str], str]) -> dict[str, Any]:
    """The wire dict of an engine or host event. Images become metadata only; free text goes through `redact`. An
    event with no wire form (the workflow events, which only the local app draws) is `WireError`.

    `redact` is `Callable[[str], str]`. The engine's `Redactor.redact` returns `(text, count)`, so the caller adapts
    it (e.g. `lambda t: redactor.redact(t)[0]`). If it raises, the result is `WireError("redaction failed")`:
    there is no unredacted fallback and the original message is not kept.
    """

    def safe(text: str) -> str:
        try:
            return redact(text)
        except Exception:
            raise WireError("redaction failed") from None

    head: dict[str, Any] = {"session_id": session_id, "seq": seq}
    return _build(event, head, safe).model_dump(mode="json")


def _event(message: wm.Message) -> WireEvent:
    match message:
        case wm.ProviderState():
            return hev.ProviderState(message.problem)
        case wm.ModelChanged():
            return hev.ModelChanged(message.model, message.window)
        case wm.QueueChanged():
            # A remote client shows the typed text; what is sent stays on the daemon's side.
            return hev.QueueChanged(tuple(Queued(e.typed, e.typed, e.label or None, e.client) for e in message.entries))
        case wm.ModeChanged():
            return hev.ModeChanged(Mode(message.mode))
        case wm.SessionStarting():
            return hev.SessionStarting(message.active)
        case wm.TurnUndone():
            return hev.TurnUndone(message.client)
        case wm.UserEntry():
            return hev.UserEntry(message.text, message.label, message.client)
        case wm.Delivering():
            return hev.Delivering()
        case wm.JobEventsDone():
            return hev.JobEventsDone()
        case wm.EntryEnd():
            return hev.EntryEnd(message.discard)
        case wm.TurnOutcome():
            check = message.init_check
            return hev.TurnOutcome(
                message.kind,
                message.status,
                continued=message.continued,
                restore_text=message.restore_text,
                queue_restore=message.queue_restore,
                delivered=message.delivered,
                turn=message.turn,
                init_check=None if check is None else (check.level, check.text),
                ack=message.ack,
                client=message.client,
                interrupted_by=message.interrupted_by,
                restores=tuple((r.client, r.text) for r in message.restores),
            )
        case wm.AssistantText():
            return ev.AssistantText(message.delta)
        case wm.ToolStarted():
            return ev.ToolStarted(message.call_id, message.label, message.task, message.name, message.args)
        case wm.ToolFinished():
            result = ToolResult(
                output=message.output, is_error=message.is_error, summary=message.summary, display=message.display
            )
            preview = message.preview
            return ev.ToolFinished(
                message.call_id,
                result,
                Preview(preview.kind, preview.title, preview.body) if preview else None,
                message.applied,
            )
        case wm.Notice():
            return ev.Notice(message.level, message.text)
        case wm.UsageUpdate():
            return ev.UsageUpdate(
                message.used, message.window, message.estimated, message.prompt_tokens, message.completion_tokens
            )
        case wm.Compacting():
            return ev.Compacting(message.active)
        case wm.TurnEnded():
            return ev.TurnEnded(message.status)
        case wm.TodosChanged():
            return ev.TodosChanged(tuple(TodoItem(i.content, i.status) for i in message.items))
        case wm.TaskProgress():
            # a nested event is one of the engine's (the model's `WireEvent` has no host arms)
            return ev.TaskProgress(message.call_id, cast(ev.AgentEvent, _event(message.event)))
        case wm.ToolProgress():
            return ev.ToolProgress(message.text)
        case wm.Phase():
            return ev.Phase(message.kind, message.step, message.max_steps, message.detail)
    raise WireError(f"{message.type} is not an engine event")


def trust_to_wire(ask: TrustAsk) -> wm.TrustAsk:
    """A trust prompt as the wire carries it (its texts are built and escaped by `core.trust`, so nothing is added)."""
    return wm.TrustAsk(
        kind=ask.kind,
        lines=list(ask.lines),
        note=ask.note,
        title=ask.title,
        header=list(ask.header),
        footer=list(ask.footer),
        decline=ask.decline,
        accept=ask.accept,
    )


def trust_from_wire(ask: wm.TrustAsk) -> TrustAsk:
    return TrustAsk(
        ask.kind,
        list(ask.lines),
        ask.note,
        title=ask.title,
        header=list(ask.header),
        footer=list(ask.footer),
        decline=ask.decline,
        accept=ask.accept,
    )


def from_wire(data: dict[str, Any]) -> WireEvent | wm.Message:
    """An engine event for event messages, the wire model for the rest (snapshot, ask, ...)."""
    if not isinstance(data, dict):
        raise WireError("a wire message must be an object")
    version = data.get("v")
    if type(version) is not int or version != wm.PROTOCOL_VERSION:
        raise WireError("unsupported protocol version")
    kind = data.get("type")
    if not isinstance(kind, str) or kind not in _KNOWN:
        raise WireError("unknown message type")
    try:
        message: wm.Message = _ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise WireError(f"bad {kind} message: {exc.error_count()} errors") from exc
    try:
        return _event(message)
    except WireError:
        return message


def validate_answer(ask: wm.Ask, answer: wm.Answer) -> None:
    """Raise `WireError` unless `answer` is one the user could have given to `ask`."""
    request = ask.request
    if isinstance(request, wm.TrustAsk):
        if not isinstance(answer, wm.TrustAnswer):
            raise WireError("a trust ask takes a trust answer")
        return
    if not isinstance(answer, wm.DecisionAnswer):
        raise WireError("an approval ask takes a decision answer")
    choice = answer.choice
    if request.protected and choice not in (Choice.ONCE, Choice.DENY):
        raise WireError("a protected path allows only once or deny")
    if choice is Choice.ALWAYS and not request.allow_always:
        raise WireError("always is not offered")
    if choice is Choice.PROJECT and request.save_rule is None:
        raise WireError("always here is not offered")


__all__ = ["WireError", "WireEvent", "from_wire", "to_wire", "trust_from_wire", "trust_to_wire", "validate_answer"]
