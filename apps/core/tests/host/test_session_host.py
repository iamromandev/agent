"""SessionHost: the turn pipeline, its claims and the queue, with no UI (a test client pumps the bus)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from core.engine import Agent
from core.engine.approval import Approver
from core.engine.events import AssistantText, Notice, TurnEnded
from core.engine.hooks.setup import HookDecision
from core.engine.mode import Mode
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import FileRecorder, SessionStore, sessions_root, workspace_dir
from core.engine.session.lock import LOCKED_HINT
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.host.asks import AskEvent
from core.host.events import EntryEnd, TurnOutcome, UserEntry
from core.host.queue import Queued
from core.host.session import SessionHost
from core.testing.fakes import FakeCatalog, Gate, ScriptedProvider
from core.testing.helpers import session_held_elsewhere

REPLY = [TextDelta("ok"), Done(1, 1)]


class Client:
    """Draws nothing: it keeps every event, marks it handled, and can hold one kind of event to model a slow draw."""

    def __init__(self, host: SessionHost) -> None:
        self.host = host
        self.events: list[object] = []
        self.hold: type | None = None
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.sub = host.bus.subscribe(queue_size=0)
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            _, event = await self.sub.queue.get()
            self.events.append(event)
            if self.hold is not None and isinstance(event, self.hold):
                self.held.set()
                await self.release.wait()
            if isinstance(event, TurnOutcome):
                self.host.outcome_applied(event.ack)  # applied: the host may start the next job
            self.sub.queue.task_done()

    def outcomes(self) -> list[TurnOutcome]:
        return [e for e in self.events if isinstance(e, TurnOutcome)]

    def notices(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, Notice)]


def _host(tmp_path: Path, script: list[list[ProviderEvent | Gate]], *, models: tuple[str, ...] = ("fake",)) -> SessionHost:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    root = ws.resolve()
    provider = ScriptedProvider(script)
    sessions = sessions_root(None)

    def factory(approver: Approver) -> Agent:
        return Agent(provider, default_tools(), approver, ToolContext(root, 5.0), 5, recorder=FileRecorder(workspace_dir(sessions, root), root))

    return SessionHost(
        factory,
        provider,
        root,
        window_cap=8192,
        store=SessionStore(sessions, root),
        recorder_factory=lambda: FileRecorder(workspace_dir(sessions, root), root),
        catalog=FakeCatalog(list(models)),
    )


@pytest.fixture
def made(tmp_path: Path):
    def build(script, **kw) -> tuple[SessionHost, Client]:
        host = _host(tmp_path, script, **kw)
        return host, Client(host)

    return build


async def test_a_turn_publishes_the_entry_the_reply_and_its_outcome(made) -> None:
    host, client = made([REPLY])
    assert host.send("hi").status == "started" and host.turn_active
    await host.idle()

    assert not host.turn_active
    kinds = [type(e).__name__ for e in client.events]
    assert kinds.index("UserEntry") < kinds.index("AssistantText") < kinds.index("TurnEnded") < kinds.index("TurnOutcome")
    (outcome,) = client.outcomes()
    assert outcome.status == "done" and outcome.delivered and outcome.restore is None
    assert EntryEnd(discard=False) in client.events


async def test_messages_sent_during_a_turn_are_queued_and_sent_as_one_when_it_ends_done(made) -> None:
    gate = Gate()
    host, client = made([[gate, *REPLY], REPLY])
    host.send("first")
    await asyncio.wait_for(gate.reached.wait(), 5)
    assert host.send("a").status == "queued" and host.send("b").status == "queued"
    assert host.send("   ").status == "ignored"
    gate.release.set()
    await host.idle()

    assert [o.continued for o in client.outcomes()] == [True, False]
    assert [e.text for e in client.events if isinstance(e, UserEntry)] == ["first", "a\n\nb"]
    assert not host.queue


async def test_an_interrupted_turn_gives_the_queue_back_instead_of_sending_it(made) -> None:
    gate = Gate()
    host, client = made([[gate, *REPLY], REPLY])
    host.send("first")
    await asyncio.wait_for(gate.reached.wait(), 5)
    host.send("queued one")
    host.interrupt()
    await host.idle()

    (outcome,) = client.outcomes()
    assert outcome.status == "interrupted" and not outcome.continued
    assert outcome.restore == "queued one"  # the first message reached the agent: only the queue comes back


async def test_an_interrupt_before_the_turn_entered_ends_it_at_once_with_the_draft(made) -> None:
    host, client = made([REPLY])
    host.queue.add(Queued("q", "q", None))
    host.send("x")
    host.interrupt()  # no await in between: the task never ran
    assert not host.queue  # ended at once: the queue went into its outcome
    await host.idle()

    assert not host.turn_active  # released once the client applied the outcome
    (outcome,) = client.outcomes()
    assert outcome.restore == "x\nq" and not outcome.delivered
    assert host.provider.requests == []


async def test_an_interrupt_while_the_entry_is_drawn_keeps_the_message_out_and_discards_the_entry(made) -> None:
    host, client = made([REPLY])
    client.hold = UserEntry
    host.send("second")
    await asyncio.wait_for(client.held.wait(), 5)
    host.interrupt()
    client.release.set()
    await host.idle()

    assert EntryEnd(discard=True) in client.events
    (outcome,) = client.outcomes()
    assert outcome.restore == "second" and not outcome.delivered
    assert host.provider.requests == []


async def test_a_message_sent_before_the_client_applied_the_outcome_waits_and_never_gets_the_old_draft(made) -> None:
    host, client = made([REPLY])
    client.hold = TurnOutcome  # the client is slow to apply the end of the turn
    host.send("first", draft="first typed")
    host.interrupt()  # before it entered: the draft goes back to the box
    await asyncio.wait_for(client.held.wait(), 5)

    second = host.send("second")  # typed while the box still looks busy
    held = host.turn_active
    client.release.set()  # before asserting: a held client would leave the host waiting at teardown
    await asyncio.wait_for(host.idle(), 5)

    assert second.status == "queued" and held  # no turn starts before the old outcome is applied
    first, *rest = client.outcomes()
    assert first.restore == "first typed"  # given back to the box before anything new started
    assert [e.text for e in client.events if isinstance(e, UserEntry)] == ["second"]
    assert [o.restore for o in rest] == [None] and not host.turn_active


async def test_an_interrupt_before_the_outcome_is_applied_gives_back_what_was_queued_meanwhile(made) -> None:
    host, client = made([REPLY])
    client.hold = TurnOutcome
    host.send("first")
    host.interrupt()
    await asyncio.wait_for(client.held.wait(), 5)
    host.send("second")
    host.interrupt()  # Esc while it still looked busy: "second" must not be sent
    client.release.set()
    await asyncio.wait_for(host.idle(), 5)

    assert [o.restore for o in client.outcomes()] == ["first", "second"]
    assert not any(isinstance(e, UserEntry) for e in client.events) and not host.turn_active


async def test_an_outcome_no_client_applies_releases_the_claim_after_a_bounded_wait(made, monkeypatch) -> None:
    monkeypatch.setattr("core.host.session.OUTCOME_ACK_TIMEOUT", 0.05)
    host, client = made([REPLY])
    client.hold = TurnOutcome  # a remote client that never acks
    host.send("x")
    host.interrupt()
    await asyncio.wait_for(client.held.wait(), 5)
    held = host.turn_active
    await asyncio.sleep(0.2)
    released = not host.turn_active
    client.release.set()
    await asyncio.wait_for(host.idle(), 5)

    assert held and released


async def test_a_failed_provider_recheck_gives_the_draft_back(made) -> None:
    host, client = made([REPLY])
    host.not_ready = True
    host.provider.check_result = "down"
    host.send("x", draft="typed x")
    await host.idle()

    assert "down" not in client.notices()  # the problem is a ProviderState, shown by the client as an error
    (outcome,) = client.outcomes()
    assert outcome.restore == "typed x" and outcome.status is None


async def test_a_blocking_prompt_hook_drops_the_message_and_gives_it_back(made, monkeypatch) -> None:
    host, client = made([REPLY])

    async def block(text, emit):
        emit(Notice("info", "hook UserPromptSubmit (global) blocked: no"))
        return HookDecision(True, "no", [], [])

    monkeypatch.setattr(host.agent, "prompt_hooks", block)
    host.send("hello")
    await host.idle()

    assert "hook UserPromptSubmit (global) blocked: no" in client.notices()
    assert not any(isinstance(e, UserEntry) for e in client.events)
    assert client.outcomes()[0].restore == "hello"


async def test_send_is_refused_while_a_session_loads_and_init_never_queues(made) -> None:
    host, _ = made([REPLY])
    assert host.begin_switch()
    refused = host.send("x")
    assert refused.status == "refused" and refused.reason == "busy — a session is loading"
    host.end_switch()
    gate = Gate()
    host.provider.script = [[gate, *REPLY]]
    host.send("first")
    await asyncio.wait_for(gate.reached.wait(), 5)
    init = host.send("prompt", init=True, draft="/init")
    assert init.status == "refused" and init.reason == "busy — press Esc to interrupt, then /init"
    gate.release.set()
    await host.idle()


async def test_undo_gives_back_the_prompt_and_holds_the_claim_meanwhile(made, monkeypatch) -> None:
    host, _ = made([REPLY])
    host.send("hello")
    await host.idle()
    seen: list[bool] = []
    real = host.agent.undo

    async def undo(emit):
        seen.append(host.turn_active and host.undoing)
        host.interrupt()  # an undo is not a turn: nothing to cancel
        return await real(emit)

    monkeypatch.setattr(host.agent, "undo", undo)
    result = await host.undo()

    assert result.prompt == "hello" and seen == [True]
    assert not host.turn_active and not host.undoing and host.agent.history == []


async def test_compact_is_a_job_in_the_turn_claim_that_ends_done(made) -> None:
    host, client = made([REPLY, REPLY, [TextDelta("summary"), Done(1, 1)]])
    host.send("one")
    await host.idle()
    host.send("two")
    await host.idle()
    assert host.compact().status == "started" and host.turn_active
    assert host.compact().status == "refused"
    await host.idle()

    assert client.outcomes()[-1].kind == "compact" and client.outcomes()[-1].status == "done"


async def test_a_model_switch_returns_its_notices_and_switches_the_provider(made) -> None:
    host, _ = made([REPLY], models=("fake", "other:7b"))
    result = await host.switch_model("other:7b")
    assert result.model == "other:7b" and host.provider.model == "other:7b"
    assert result.notices[0][0] == "info" and "other:7b" in result.notices[0][1]
    missing = await host.switch_model("nope")
    assert missing.model is None and missing.notices[0][1].startswith("Model nope is not installed")


async def test_new_and_open_session(made) -> None:
    host, _ = made([REPLY, REPLY])
    host.send("remember me")
    await host.idle()
    (info,) = await host.list_sessions()

    assert await host.new_session() is None and host.switching  # held until the client has redrawn
    host.end_switch()
    assert host.agent.history == []
    assert host.begin_switch()
    opened = await host.open_session(info)
    host.end_switch()
    assert opened.loaded is not None and host.agent.history[0].content == "remember me"


async def test_new_session_is_refused_during_a_turn(made) -> None:
    gate = Gate()
    host, _ = made([[gate, *REPLY]])
    host.send("x")
    await asyncio.wait_for(gate.reached.wait(), 5)
    assert await host.new_session("/clear") == "busy — press Esc to interrupt, then /clear"
    gate.release.set()
    await host.idle()


async def test_init_reports_when_agents_md_did_not_change(made) -> None:
    host, client = made([REPLY])
    host.send(host.init_prompt(), init=True, draft="/init")
    await host.idle()

    assert client.outcomes()[0].init_check == ("info", "/init finished without changing AGENTS.md")
    assert "/init finished without changing AGENTS.md" in client.notices()


async def test_init_checks_a_written_agents_md(made) -> None:
    write = ToolCallEvent(ToolCall("w1", "write", {"path": "AGENTS.md", "content": "`make nothing-here`\n"}))
    host, client = made([[write, Done(1, 1)], REPLY])
    host.agent.set_mode(Mode.EDITS)  # no dialog for the write
    host.send(host.init_prompt(), init=True, draft="/init")
    await host.idle()

    check = client.outcomes()[0].init_check
    assert check is not None and check[1].startswith("/init check")


async def test_a_restored_draft_and_queue_go_only_to_the_client_that_sent_them(made) -> None:
    gate = Gate()
    host, client = made([[gate, *REPLY], REPLY])
    host.send("first", client="A")
    await asyncio.wait_for(gate.reached.wait(), 5)
    host.send("from a", client="A")
    host.send("from b", client="B")
    host.interrupt(client="B")
    await host.idle()

    (entry,) = [e for e in client.events if isinstance(e, UserEntry)]
    (outcome,) = client.outcomes()
    assert entry.client == "A" and outcome.client == "A" and outcome.interrupted_by == "B"
    assert outcome.restore_for("A") == "from a" and outcome.restore_for("B") == "from b"
    assert outcome.restore_for("C") is None


async def test_a_switch_claim_is_the_clients_and_goes_when_it_disconnects(made) -> None:
    host, _ = made([REPLY], models=("fake", "other:7b"))
    assert host.begin_switch("A")
    refused = await host.switch_model("other:7b", client="B")
    assert refused.model is None and refused.notices == [("warn", "busy — a session is loading")]
    host.end_switch("B")  # not B's: nothing happens
    assert host.switching
    host.client_gone("A")
    assert not host.switching

    switched = await host.switch_model("other:7b", client="B")  # takes the claim itself, and lets it go
    assert switched.model == "other:7b" and not host.switching


async def test_open_session_takes_the_claim_itself_and_refuses_another_clients(made) -> None:
    host, _ = made([REPLY])
    host.send("remember me")
    await host.idle()
    (info,) = await host.list_sessions()
    assert await host.new_session(client="A") is None
    refused = await host.open_session(info, client="B")
    assert refused.loaded is None and refused.notices == [("warn", "busy — a session is loading")]
    host.end_switch("A")

    opened = await host.open_session(info, client="B")
    assert opened.loaded is not None and not host.switching


async def test_a_session_open_in_another_process_is_not_opened(made) -> None:
    host, _ = made([REPLY, REPLY])
    host.send("remember me")
    await host.idle()
    (info,) = await host.list_sessions()
    assert await host.new_session() is None  # the old recorder (and its lock) is closed
    host.end_switch()

    with session_held_elsewhere(info.path.parent, info.id):
        opened = await host.open_session(info)
    assert opened.loaded is None and host.agent.history == []
    assert opened.notices == [("error", f"session {info.id} is open in another agent process{LOCKED_HINT}")]


async def test_an_outcome_whose_client_disconnects_releases_the_claim_at_once(made) -> None:
    host, client = made([REPLY])
    client.hold = TurnOutcome  # A never applies it
    host.send("x", client="A")
    host.interrupt()
    await asyncio.wait_for(client.held.wait(), 5)
    held = host.turn_active
    host.client_gone("A")
    released = not host.turn_active
    client.release.set()
    await asyncio.wait_for(host.idle(), 5)

    assert held and released


async def test_an_outcome_for_a_client_that_is_not_connected_holds_nothing(made) -> None:
    host, client = made([REPLY])
    host.connected = lambda name: name != "gone"
    client.hold = TurnOutcome  # no client applies it
    host.send("x", client="gone")
    await asyncio.wait_for(client.held.wait(), 5)
    released = not host.turn_active
    client.release.set()
    await asyncio.wait_for(host.idle(), 5)

    assert released and client.outcomes()[0].ack == 0


async def test_aclose_waits_for_the_cancelled_turn(made) -> None:
    gate = Gate()
    host, _ = made([[gate, *REPLY]])
    host.send("x")
    await asyncio.wait_for(gate.reached.wait(), 5)
    await host.aclose()
    assert host.busy_tasks() == [] and host.closing


async def test_an_approval_becomes_an_ask_on_the_bus(made) -> None:
    host, client = made([[ToolCallEvent(ToolCall("b1", "bash", {"command": "true"})), Done(1, 1)], REPLY])
    host.send("run it")
    while not host.asks.pending():
        await asyncio.sleep(0.01)
    (ask,) = host.asks.pending()
    assert any(isinstance(e, AskEvent) and e.ask_id == ask.ask_id for e in client.events)
    host.interrupt()  # the ask is withdrawn with the turn
    await host.idle()
    assert host.asks.pending() == [] and client.outcomes()[0].status == "interrupted"
    assert any(isinstance(e, TurnEnded) for e in client.events)
    assert not any(isinstance(e, AssistantText) for e in client.events)
