"""`run_headless`: one agent turn without Textual; the final reply on stdout, progress on stderr, an exit code."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from src.engine import Agent
from src.engine.commands.builtins import BUILTINS
from src.engine.compaction import STOP_NOTE_PREFIX
from src.engine.events import AgentEvent, Notice, TurnEnded
from src.engine.loop import close_recorder
from src.engine.permissions.rules import Rule, RuleError, parse_rule
from src.engine.permissions.setup import run_rule_warnings, tool_keys
from src.engine.provider.base import SwitchableProvider
from src.engine.session import FileRecorder, SessionError, SessionWriteError
from src.engine.text import escape_display
from src.engine.tools import default_tools
from src.headless import EXIT_FAIL, EXIT_OK, EXIT_USAGE, HeadlessOptions, UsageError
from src.headless.approver import HeadlessApprover
from src.headless.output import StderrSink, want_color, write_stdout
from src.runtime import Runtime, build_runtime


def run_headless(
    workspace: Path, opts: HeadlessOptions, *, provider: SwitchableProvider | None = None
) -> int:
    """Run one turn for `opts` in `workspace` and return the exit code. `provider` replaces Ollama (for tests)."""
    sink = StderrSink(sys.stderr, opts.quiet, want_color(sys.stderr))
    rt: Runtime | None = None
    started: list[bool] = []  # set once the loop that owns the runtime's connections has taken over
    try:
        allow = _parse_allow(opts.allow)
        prompt = (opts.prompt or "").strip()
        if not prompt:
            raise UsageError("give a prompt: agent -p \"<prompt>\"")
        rt = build_runtime(workspace, BUILTINS, model=opts.model, terminals=False, provider=provider)
        rt.permissions.rules.add_run_allow(allow)
        tools = default_tools(None, fetcher=rt.web, searcher=rt.search, skills=rt.skills)
        warnings = run_rule_warnings(allow, tool_keys(tools))
        for warning in (*rt.warnings, *warnings, *rt.commands.warnings, *rt.skills.warnings):
            sink(Notice("warn", warning))
        for note in (*rt.model_notices, *rt.guard.info):
            sink(Notice("info", note))
        return asyncio.run(_run(rt, opts, prompt, sink, started))
    except UsageError as exc:
        print(f"agent: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # never a traceback: one line and exit 1
        print(f"agent: internal error: {escape_display(str(exc))}", file=sys.stderr)
        return EXIT_FAIL
    finally:
        if rt is not None:
            if not started:
                rt.abort()  # the loop never ran, so nothing closed the clients
            rt.kill_all()  # after `close` ran in the loop, this only covers what it could not stop


def _parse_allow(texts: tuple[str, ...]) -> list[Rule]:
    rules: list[Rule] = []
    for text in texts:
        try:
            rules.append(parse_rule(text))
        except RuleError as exc:
            raise UsageError(f"invalid --allow rule {text!r}: {exc}") from exc
    return rules


async def _run(
    rt: Runtime, opts: HeadlessOptions, prompt: str, sink: Callable[[AgentEvent], None], started: list[bool]
) -> int:
    started.append(True)
    status: list[str] = []

    def emit(event: AgentEvent) -> None:
        if isinstance(event, TurnEnded):
            status.append(event.status)
        sink(event)

    try:
        agent = rt.agent_factory(HeadlessApprover())
        try:
            if (problem := await rt.provider.check()) is not None:
                sink(Notice("error", problem))
                return EXIT_FAIL
            source = await _open_session(rt, agent, opts, sink)
            await agent.session_started(source, sink)
            decision = await agent.prompt_hooks(prompt, sink)
            if decision.blocked:
                return EXIT_FAIL  # the hook's line, with its reason, is already on stderr
            await agent.run_turn(prompt, emit, context=decision.context)
        except UsageError:
            raise
        except Exception as exc:  # an internal error is exit 1 with the message, never a traceback
            sink(Notice("error", f"internal error: {exc}"))
            return EXIT_FAIL
        finally:
            close_recorder(agent.recorder)  # flush what is queued before the id is reported
        done = status[-1:] == ["done"]
        if done:
            write_stdout(_final_text(agent))
        session_id = getattr(agent.recorder, "session_id", "")
        if isinstance(session_id, str) and session_id:
            sink(Notice("info", f"session {session_id} saved"))
        return EXIT_OK if done else EXIT_FAIL
    finally:
        await rt.close()


def _final_text(agent: Agent) -> str:
    """The text of the last assistant message of this turn (after its prompt; a Stop continuation note is not a
    prompt), or "" when the turn ended without text: an earlier turn's reply is never printed."""
    history = agent.history
    start = next(
        (i for i in range(len(history) - 1, -1, -1) if history[i].role == "user" and not history[i].content.startswith(STOP_NOTE_PREFIX)),
        -1,
    )
    return next((m.content for m in reversed(history[start + 1 :]) if m.role == "assistant"), "")


async def _open_session(
    rt: Runtime, agent: Agent, opts: HeadlessOptions, sink: Callable[[AgentEvent], None]
) -> Literal["startup", "resume"]:
    """Make the session the options ask for the live one; the `SessionStart` source (`startup` or `resume`)."""
    kind = opts.start.kind
    if kind == "new":
        return "startup"
    try:
        if kind == "continue":
            info = await asyncio.to_thread(rt.store.latest)
            if info is None:
                sink(Notice("info", "No previous session for this folder"))
                return "startup"
        else:
            info = await asyncio.to_thread(rt.store.find, opts.start.session_id or "")
        loaded = await asyncio.to_thread(rt.store.load, info.id)
        recorder = FileRecorder.resume(loaded.info.path, rt.root)
    except (SessionError, SessionWriteError) as exc:
        raise UsageError(str(exc)) from exc
    old = agent.reset(recorder=recorder, history=loaded.history, tool_meta=loaded.tool_meta, todos=loaded.todos)
    if old is not None:
        await asyncio.to_thread(close_recorder, old)
    return "resume"
