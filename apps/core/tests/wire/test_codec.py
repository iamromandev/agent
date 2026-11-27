import json

import pytest
from core.engine import events as ev
from core.engine.configguard import NEXT_LAUNCH, HeldGlobal
from core.engine.permissions.rules import parse_rule
from core.engine.provider.base import Image
from core.engine.todo import TodoItem
from core.engine.tools.base import Preview, ToolResult
from core.host import events as hev
from core.trust import trust_lines
from core.wire import models as wm
from core.wire.codec import WireError, from_wire, to_wire, trust_from_wire, trust_to_wire, validate_answer


def same(text: str) -> str:
    return text


def mask(text: str) -> str:
    return text.replace("hunter2hunter2", "[redacted:env:X]")


EVENTS: list[ev.AgentEvent] = [
    ev.AssistantText("hi"),
    ev.ToolStarted("c1", "bash ls", True, "bash", {"command": "ls", "n": [1, 2]}),
    ev.ToolFinished("c1", ToolResult("out", True, "exit 1", "\x1b[1mout"), Preview("diff", "edit a", "+x"), True),
    ev.ToolFinished("c2", ToolResult("out"), None, False),
    ev.Notice("warn", "careful"),
    ev.UsageUpdate(10, 100, True, 3, 4),
    ev.Compacting(True),
    ev.TurnEnded("step_limit"),
    ev.TodosChanged((TodoItem("a", "done"), TodoItem("b", "pending"))),
    ev.TaskProgress("c1", ev.Notice("info", "child")),
    ev.TaskProgress("c1", ev.ToolFinished("c9", ToolResult("o"), None, True)),
    ev.ToolProgress("step 1/9"),
    ev.Phase("running", 2, 50, "bash"),
]


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: type(e).__name__)
def test_every_event_round_trips(event: ev.AgentEvent) -> None:
    data = to_wire(event, session_id="s1", seq=7, redact=same)
    assert data["v"] == 1 and data["session_id"] == "s1" and data["seq"] == 7
    assert from_wire(json.loads(json.dumps(data))) == event


def test_image_bytes_are_never_on_the_wire() -> None:
    result = ToolResult("img", images=(Image("png", b"SECRETPIXELS", "a.png"),))
    data = to_wire(ev.ToolFinished("c", result, None, True), session_id=None, seq=1, redact=same)
    assert "SECRETPIXELS" not in json.dumps(data)
    assert data["images"] == [{"format": "png", "name": "a.png", "size": 12}]


def test_a_secret_in_a_notice_is_redacted() -> None:
    data = to_wire(ev.Notice("info", "key hunter2hunter2"), session_id=None, seq=1, redact=mask)
    assert "hunter2hunter2" not in json.dumps(data)


def test_tool_args_are_redacted() -> None:
    event = ev.ToolStarted("c", "l", False, "bash", {"command": "echo hunter2hunter2", "x": ["hunter2hunter2"]})
    data = to_wire(event, session_id=None, seq=1, redact=mask)
    assert "hunter2hunter2" not in json.dumps(data)


@pytest.mark.parametrize(
    "patch", [{"type": "nope"}, {"v": 2}, {"v": None}, {"level": "loud"}], ids=["type", "version", "no-version", "shape"]
)
def test_unknown_type_version_or_shape_raises(patch: dict[str, object]) -> None:
    data = to_wire(ev.Notice("info", "x"), session_id=None, seq=1, redact=same) | patch
    with pytest.raises(WireError):
        from_wire(data)


def test_non_event_messages_come_back_as_models() -> None:
    snap = wm.Snapshot(session_id="s", model="m", records=[{"a": 1}])
    assert from_wire(snap.model_dump(mode="json")) == snap
    ask = wm.Ask(
        ask_id="a1", kind="mcp_trust", request=wm.TrustAsk(kind="mcp_trust", lines=["x"], note=None)
    )
    assert from_wire(ask.model_dump(mode="json")) == ask
    resolved = wm.AskResolved(ask_id="a1", by="c", decision=wm.TrustAnswer(trust=True))
    assert from_wire(resolved.model_dump(mode="json")) == resolved


@pytest.mark.parametrize(
    "data",
    [[], "x", None, {"v": 1, "type": ["a"]}, {"v": 1, "type": {}}, {"v": True, "type": "notice"}, {"v": 1.0, "type": "notice"}],
    ids=["list", "str", "none", "type-list", "type-dict", "bool-version", "float-version"],
)
def test_malformed_input_raises_wire_error_only(data: object) -> None:
    with pytest.raises(WireError):
        from_wire(data)  # ty: ignore[invalid-argument-type]


def test_summary_and_arg_keys_are_redacted() -> None:
    finished = ev.ToolFinished("c", ToolResult("o", summary="exit hunter2hunter2"), None, True)
    assert "hunter2hunter2" not in json.dumps(to_wire(finished, session_id=None, seq=1, redact=mask))
    started = ev.ToolStarted("c", "l", False, "bash", {"hunter2hunter2": {"hunter2hunter2": 1}})
    assert "hunter2hunter2" not in json.dumps(to_wire(started, session_id=None, seq=1, redact=mask))


def test_a_failing_redactor_is_a_wire_error_without_the_message() -> None:
    def boom(text: str) -> str:
        raise RuntimeError("secret-detail")

    with pytest.raises(WireError, match="redaction failed") as info:
        to_wire(ev.Notice("info", "x"), session_id=None, seq=1, redact=boom)
    assert "secret-detail" not in str(info.value)


HOST_EVENTS: list[object] = [
    hev.ProviderState("Ollama is down"),
    hev.ProviderState(None),
    hev.UserEntry("hi", "/x (global)", "A"),
    hev.Delivering(),
    hev.JobEventsDone(),
    hev.EntryEnd(True),
    hev.TurnOutcome(
        "turn",
        "interrupted",
        restore_text="x",
        queue_restore="q",
        turn=3,
        init_check=("info", "/init done"),
        ack=2,
        client="A",
        interrupted_by="B",
        restores=(("A", "x"), ("B", "q")),
    ),
]


@pytest.mark.parametrize("event", HOST_EVENTS, ids=lambda e: type(e).__name__)
def test_every_host_event_round_trips(event: object) -> None:
    data = to_wire(event, session_id="s1", seq=2, redact=same)  # ty: ignore[invalid-argument-type]
    assert from_wire(json.loads(json.dumps(data))) == event


def test_a_trust_ask_round_trips_with_its_texts() -> None:
    ask = trust_lines("permission_trust", rules=[parse_rule("bash(ls*)")])
    assert ask.title and ask.header and ask.footer and ask.decline and ask.accept
    assert trust_from_wire(trust_to_wire(ask)) == ask


def test_the_global_ask_carries_the_empty_and_next_launch_texts() -> None:
    held = HeldGlobal(commands=("c",))
    empty = trust_lines("global_trust", held=HeldGlobal())
    full = trust_lines("global_trust", held=held)
    assert "It holds no allow rules, hooks or MCP servers now." in trust_to_wire(empty).header
    assert any(NEXT_LAUNCH in line for line in trust_to_wire(full).footer)


def test_a_withdrawn_ask_has_no_decision() -> None:
    resolved = wm.AskResolved(ask_id="a1", by="", decision=None)
    assert from_wire(resolved.model_dump(mode="json")) == resolved


def test_a_workflow_event_has_no_wire_form() -> None:
    with pytest.raises(WireError):
        to_wire(ev.WorkflowLine("info", "x"), session_id=None, seq=1, redact=same)


def _trust(kind: str = "mcp_trust") -> wm.Ask:
    return wm.Ask(ask_id="a", kind=kind, request=wm.TrustAsk(kind=kind, lines=[]))  # ty: ignore[invalid-argument-type]


def _approval(**kw: object) -> wm.Ask:
    request = wm.ApprovalRequest(tool="bash", label="l", allow_always=True, **kw)  # ty: ignore[invalid-argument-type]
    kind = "protected" if request.protected else "approval"
    return wm.Ask(ask_id="a", kind=kind, request=request)


def test_ask_kind_must_match_request() -> None:
    with pytest.raises(ValueError):
        wm.Ask(ask_id="a", kind="approval", request=wm.TrustAsk(kind="mcp_trust", lines=[]))
    with pytest.raises(ValueError):
        wm.Ask(ask_id="a", kind="mcp_trust", request=wm.TrustAsk(kind="global_trust", lines=[]))
    with pytest.raises(ValueError):
        wm.Ask(ask_id="a", kind="mcp_trust", request=wm.ApprovalRequest(tool="t", label="l", allow_always=True))
    with pytest.raises(ValueError):
        wm.Ask(ask_id="a", kind="protected", request=wm.ApprovalRequest(tool="t", label="l", allow_always=False))
    with pytest.raises(ValueError):
        wm.Ask(ask_id="a", kind="approval", request=wm.ApprovalRequest(tool="t", label="l", allow_always=True, protected=True))


def test_nested_models_forbid_extra_keys() -> None:
    with pytest.raises(ValueError):
        wm.PreviewModel(kind="diff", title="t", body="b", x=1)  # ty: ignore[unknown-argument]


def _decision(choice: str) -> wm.DecisionAnswer:
    return wm.DecisionAnswer(choice=choice)


def test_answers_must_fit_the_ask() -> None:
    validate_answer(_trust(), wm.TrustAnswer(trust=True))
    with pytest.raises(WireError):
        validate_answer(_trust(), _decision("once"))
    with pytest.raises(WireError):
        validate_answer(_approval(), wm.TrustAnswer(trust=True))
    validate_answer(_approval(), _decision("once"))
    validate_answer(_approval(), _decision("always"))


def test_always_needs_allow_always_and_project_needs_save_rule() -> None:
    no_always = wm.Ask(
        ask_id="a", kind="approval", request=wm.ApprovalRequest(tool="t", label="l", allow_always=False)
    )
    with pytest.raises(WireError):
        validate_answer(no_always, _decision("always"))
    validate_answer(no_always, _decision("deny"))
    with pytest.raises(WireError):
        validate_answer(_approval(), _decision("project"))
    validate_answer(_approval(save_rule="bash(ls)"), _decision("project"))


def test_protected_allows_only_once_or_deny() -> None:
    ask = _approval(protected=True, protected_path="/x")
    validate_answer(ask, _decision("once"))
    validate_answer(ask, _decision("deny"))
    for choice in ("always", "project"):
        with pytest.raises(WireError):
            validate_answer(ask, _decision(choice))
