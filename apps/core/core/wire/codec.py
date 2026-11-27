"""Engine events to wire dicts and back."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from pydantic import TypeAdapter, ValidationError

from core.engine import events as ev
from core.engine.approval import Choice
from core.engine.todo import TodoItem
from core.engine.tools.base import Preview, ToolResult
from core.wire import models as wm

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


def _build(event: ev.AgentEvent, head: dict[str, Any], redact: Callable[[str], str]) -> wm.Message:
    match event:
        case ev.AssistantText():
            return wm.AssistantText(**head, delta=event.delta)
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


def to_wire(event: ev.AgentEvent, *, session_id: str | None, seq: int, redact: Callable[[str], str]) -> dict[str, Any]:
    """The wire dict of an engine event. Images become metadata only; free text goes through `redact`.

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


def _event(message: wm.Message) -> ev.AgentEvent:
    match message:
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
            return ev.TaskProgress(message.call_id, _event(message.event))
        case wm.ToolProgress():
            return ev.ToolProgress(message.text)
        case wm.Phase():
            return ev.Phase(message.kind, message.step, message.max_steps, message.detail)
    raise WireError(f"{message.type} is not an engine event")


def from_wire(data: dict[str, Any]) -> ev.AgentEvent | wm.Message:
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


__all__ = ["WireError", "from_wire", "to_wire", "validate_answer"]
