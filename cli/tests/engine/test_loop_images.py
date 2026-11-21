"""Images on a user turn: `run_turn(images=...)` puts them on the user message, the provider gets them, the session keeps them."""

from __future__ import annotations

from src.engine.loop import Agent
from src.engine.provider.base import Done, Image, TextDelta
from src.engine.session import NullRecorder
from src.engine.tools.base import ToolContext

from tests.engine.test_loop import WINDOW, Events
from tests.fakes import RecordingApprover, ScriptedProvider

PNG = Image("png", b"\x89PNG\r\n\x1a\nxyz", "a.png")


class _Recorder(NullRecorder):
    def __init__(self) -> None:
        self.saved: list = []

    def message(self, message, meta) -> None:
        self.saved.append(message)


def _agent(tmp_path, provider, recorder=None) -> Agent:
    return Agent(provider, [], RecordingApprover(), ToolContext(tmp_path, 5.0), 5, recorder=recorder or NullRecorder())


async def test_run_turn_puts_the_images_on_the_user_message(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    agent = _agent(tmp_path, provider)
    await agent.run_turn("see @a.png", Events(), images=(PNG,))
    sent = provider.requests[0][0]
    user = [m for m in sent if m.role == "user"][-1]
    assert user.content == "see @a.png"
    assert user.images == (PNG,)
    assert agent.history[0].images == (PNG,)


async def test_a_turn_without_images_has_none_on_any_message(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    agent = _agent(tmp_path, provider)
    await agent.run_turn("hi", Events())
    assert all(m.images == () for m in provider.requests[0][0])


async def test_the_recorder_gets_the_images(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    recorder = _Recorder()
    await _agent(tmp_path, provider, recorder).run_turn("see", Events(), images=(PNG,))
    assert recorder.saved[0].images == (PNG,)


# --- `read` on an image ------------------------------------------------------------------------------------------

from src.engine.approval import Choice, Decision  # noqa: E402
from src.engine.provider.base import ToolCall, ToolCallEvent  # noqa: E402
from src.engine.provider.vision import VisionCheck  # noqa: E402
from src.engine.tools import default_tools  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


def _read_agent(tmp_path, vision: bool | None, files=("pic.png",), decisions=None):
    for name in files:
        (tmp_path / name).write_bytes(PNG_BYTES)
    script: list = [
        [ToolCallEvent(ToolCall("c1", "read", {"path": files[0]})), Done(1, 1)],
        [TextDelta("seen"), Done(1, 1)],
    ]
    provider = ScriptedProvider(script, context_window=WINDOW, model="vis")
    check = None
    if vision is not None:
        check = VisionCheck(None)
        check.set("vis", vision)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider, default_tools(), approver, ToolContext(tmp_path, 5.0), 5, recorder=NullRecorder(), vision=check
    )
    return agent, provider, approver


async def test_read_of_a_png_returns_the_image_to_the_next_request(tmp_path) -> None:
    agent, provider, _ = _read_agent(tmp_path, True)
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content == "image pic.png (PNG, 1 KB)"
    assert [i.data for i in tool.images] == [PNG_BYTES]
    assert tool.images[0].name == "pic.png"


async def test_read_of_a_png_without_vision_is_an_error(tmp_path) -> None:
    agent, provider, _ = _read_agent(tmp_path, False)
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content == "pic.png is an image and the model 'vis' has no vision"
    assert tool.images == ()


async def test_read_of_a_png_with_no_vision_check_is_an_error(tmp_path) -> None:
    agent, provider, _ = _read_agent(tmp_path, None)
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert "has no vision" in tool.content


async def test_read_of_a_protected_png_asks_the_dialog(tmp_path) -> None:
    agent, provider, approver = _read_agent(tmp_path, True, files=("key.pem",), decisions=[Decision(Choice.DENY)])
    await agent.run_turn("look", Events())
    assert [r.protected for r in approver.requests] == [True]
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.images == ()


async def test_image_survives_redaction_truncation_and_the_session_message(tmp_path) -> None:
    agent, _, _ = _read_agent(tmp_path, True)
    await agent.run_turn("look", Events())
    tool = next(m for m in agent.history if m.role == "tool")
    assert len(tool.images) == 1


# --- lazy vision lookup, refusals and what happens to the image after the tool ---------------------------------------

import json  # noqa: E402

from src.engine.events import Notice  # noqa: E402
from src.engine.guardrails.redact import Redactor  # noqa: E402
from src.engine.guardrails.setup import REDACTION_FAILED, guard_of  # noqa: E402
from src.engine.hooks.config import DEFAULT_TIMEOUT, HookEntry  # noqa: E402
from src.engine.hooks.setup import Hooks  # noqa: E402
from src.engine.mcp.trust import TrustStore  # noqa: E402
from src.engine.permissions.setup import prepare as prepare_rules  # noqa: E402

from tests.fakes import FakeCatalog, FakeRunner, ok_run  # noqa: E402


def _catalog_agent(tmp_path, catalog: FakeCatalog, file="pic.png", content=PNG_BYTES, **kwargs):
    (tmp_path / file).write_bytes(content)
    script: list = [
        [ToolCallEvent(ToolCall("c1", "read", {"path": file})), Done(1, 1)],
        [TextDelta("seen"), Done(1, 1)],
    ]
    provider = ScriptedProvider(script, context_window=WINDOW, model="vis")
    agent = Agent(
        provider, default_tools(), RecordingApprover(), ToolContext(tmp_path, 5.0), 5,
        recorder=NullRecorder(), vision=VisionCheck(catalog), **kwargs,
    )
    return agent, provider


async def test_a_text_read_never_asks_for_capabilities(tmp_path) -> None:
    catalog = FakeCatalog(["vis"], capabilities={"vis": ("vision",)})
    agent, _ = _catalog_agent(tmp_path, catalog, "a.txt", b"hello\n")
    await agent.run_turn("look", Events())
    assert catalog.capability_calls == []


async def test_an_image_read_asks_for_capabilities_once(tmp_path) -> None:
    catalog = FakeCatalog(["vis"], capabilities={"vis": ("vision",)})
    agent, provider = _catalog_agent(tmp_path, catalog)
    await agent.run_turn("look", Events())
    assert catalog.capability_calls == ["vis"]
    assert next(m for m in provider.requests[1][0] if m.role == "tool").images != ()


async def test_a_failing_lookup_on_an_image_read_is_no_vision_plus_a_warning(tmp_path) -> None:
    catalog = FakeCatalog(["vis"], fail=True)
    agent, provider = _catalog_agent(tmp_path, catalog)
    events = Events()
    await agent.run_turn("look", events)
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert "has no vision" in tool.content and tool.images == ()
    warns = [e.text for e in events.items if isinstance(e, Notice) and e.level == "warn"]
    assert any("could not read the capabilities of 'vis'" in w for w in warns)


async def test_an_image_swapped_for_a_protected_file_after_the_path_check_is_refused(tmp_path, monkeypatch) -> None:
    agent, provider, _ = _read_agent(tmp_path, True)
    paths = guard_of(agent.ctx).paths
    monkeypatch.setattr(paths, "is_protected_fd", lambda fd: True)  # the path check passed; the open file is protected
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content == "protected path"
    assert tool.images == ()


async def test_a_deny_rule_refuses_an_image_read(tmp_path) -> None:
    settings = tmp_path / "cfg" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"deny": ["read"]}}))  # a valued rule never matches `read`
    rules = prepare_rules(tmp_path, settings, TrustStore(tmp_path / "trust.json")).rules
    agent, provider, _ = _read_agent(tmp_path, True)
    agent.rules = rules
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content.startswith("denied by rule")
    assert tool.images == ()


async def test_a_redaction_failure_drops_the_image(tmp_path, monkeypatch) -> None:
    def broken(self, text):
        raise RuntimeError("boom")

    monkeypatch.setattr(Redactor, "redact", broken)
    agent, provider, _ = _read_agent(tmp_path, True)
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content == REDACTION_FAILED
    assert tool.images == ()


async def test_posttooluse_context_keeps_the_image(tmp_path) -> None:
    agent, provider, _ = _read_agent(tmp_path, True)
    runner = FakeRunner([ok_run(json.dumps({"context": "note"}))])
    entry = HookEntry("PostToolUse", "x", None, DEFAULT_TIMEOUT, "global")
    agent.hooks = Hooks([entry], [], runner, lambda: True, tmp_path)
    await agent.run_turn("look", Events())
    tool = next(m for m in provider.requests[1][0] if m.role == "tool")
    assert tool.content.endswith("[hook PostToolUse] note")
    assert len(tool.images) == 1
