"""`run_headless`: one agent turn without Textual; the final reply on stdout, progress on stderr, an exit code."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from src.engine import Agent
from src.engine.commands.builtins import BUILTINS
from src.engine.compaction import STOP_NOTE_PREFIX
from src.engine.configguard import held_warning
from src.engine.configwatch import Cause
from src.engine.events import AgentEvent, Notice, TurnEnded
from src.engine.loop import close_recorder
from src.engine.mcp.contract import McpConnector
from src.engine.mcp.hub import HubEvent
from src.engine.mentions import Expansion, MentionNote, apply_vision, expand_mentions, note_text
from src.engine.permissions.rules import Rule, RuleError, parse_rule
from src.engine.permissions.setup import run_rule_warnings, tool_keys
from src.engine.provider.base import Image, SwitchableProvider
from src.engine.provider.models import ModelCatalog
from src.engine.session import FileRecorder, SessionError, SessionWriteError
from src.engine.text import escape_display
from src.engine.tools import default_tools
from src.headless import EXIT_FAIL, EXIT_INT, EXIT_OK, EXIT_TERM, EXIT_USAGE, HeadlessOptions, UsageError
from src.headless.approver import HeadlessApprover
from src.headless.output import JsonCollector, Outcome, StderrSink, StopReason, json_line, want_color, write_stdout
from src.headless.prompt import build_prompt
from src.runtime import Runtime, build_runtime


class _Terminated(BaseException):
    """SIGTERM before the loop took over (`_main` then handles the signal itself)."""


def _raise_terminated(number: int, frame: object) -> None:
    raise _Terminated


@dataclass
class _State:
    """What the exit paths leave for `_finish`: how the run ended, the reply and what the result object reports."""

    json_mode: bool
    collector: JsonCollector | None = None
    outcome: Outcome | None = None
    reply: str = ""
    session_id: str | None = None
    model: str | None = None
    redact: Callable[[str], str] = lambda text: text  # the runtime's redactor once there is one
    finished: bool = False
    image_notes: list[MentionNote] = field(default_factory=list)  # `attached` lines of the image mentions, shown once vision is known
    images: list[Image] = field(default_factory=list)


def _fanout(*sinks: Callable[[AgentEvent], None]) -> Callable[[AgentEvent], None]:
    def sink(event: AgentEvent) -> None:
        for each in sinks:
            each(event)

    return sink


def _finish(state: _State) -> int:
    """Write the one final JSON object (JSON modes only; text mode wrote its reply already) and return the exit code.
    Whatever goes wrong while building or writing it, a minimal ASCII result is written instead."""
    outcome = state.outcome or Outcome(EXIT_FAIL, "error", "internal error: no outcome")
    if state.json_mode and state.collector is not None and not state.finished:
        for attempt in range(3):
            try:
                if attempt == 0:
                    state.collector.flush()
                    write_stdout(json_line(state.collector.result(outcome, state.reply, state.session_id, state.model)))
                else:
                    write_stdout(_minimal(outcome))
                state.finished = True
                break
            except (Exception, KeyboardInterrupt, _Terminated):
                continue
    return outcome.exit_code


def _minimal(outcome: Outcome) -> str:
    error = outcome.error if outcome.exit_code else None
    return json.dumps(
        {
            "type": "result",
            "is_error": outcome.exit_code != 0,
            "exit_code": outcome.exit_code,
            "stop_reason": outcome.stop_reason,
            "error": error,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def run_headless(
    workspace: Path,
    opts: HeadlessOptions,
    *,
    provider: SwitchableProvider | None = None,
    connector: McpConnector | None = None,
    stdin: BinaryIO | None = None,
    catalog: ModelCatalog | None = None,
) -> int:
    """Run one turn for `opts` in `workspace` and return the exit code. `provider` replaces Ollama, `connector` the
    MCP servers', `catalog` the model catalog that answers the vision check and `stdin` the process's (bytes; all
    for tests)."""
    json_mode = opts.output != "text"
    state = _State(json_mode)
    stderr_sink = StderrSink(sys.stderr, opts.quiet or json_mode, want_color(sys.stderr))
    sink: Callable[[AgentEvent], None] = stderr_sink
    if json_mode:
        state.collector = JsonCollector(None, lambda text: state.redact(text), streaming=opts.output == "stream-json")
        sink = _fanout(stderr_sink, state.collector)
    rt: Runtime | None = None
    started: list[bool] = []  # set once the loop that owns the runtime's connections has taken over
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, _raise_terminated)  # SIGTERM while the runtime is built is 143 too
    try:
        allow = _parse_allow(opts.allow)
        rt = build_runtime(
            workspace,
            BUILTINS,
            model=opts.model,
            terminals=False,
            provider=provider,
            connector=connector,
            trust_project_mcp=opts.trust_project,
            catalog=catalog,
        )
        state.redact = _redactor(rt)
        # A change the config watch sees is a warning (kept with --quiet; a `notice` in stream-json), also one seen
        # while the runtime was built, which the guard holds until this first listener.
        rt.config_guard.listen(lambda text: sink(Notice("warn", text)))
        model = getattr(rt.provider, "model", None)
        state.model = model if isinstance(model, str) else None
        mention_notes: list[MentionNote] = []

        def attach(text: str) -> str:  # the user's own prompt only, not piped input
            expansion = expand_mentions(text, rt.root, rt.guard)
            # image lines wait for the vision check, which is async and asks about the model that runs
            mention_notes.extend(n for n in expansion.notes if n.image is None)
            state.image_notes.extend(n for n in expansion.notes if n.image is not None)
            state.images.extend(expansion.images)
            return expansion.text

        prompt = build_prompt(opts.prompt, *_stdin(stdin), rt.commands, BUILTINS, attach)
        rt.permissions.rules.add_run_allow(allow)
        if opts.trust_project:
            rt.permissions.rules.trust_for_run()  # hooks follow the rules' trust; no store is written
        tools = default_tools(None, fetcher=rt.web, searcher=rt.search, skills=rt.skills)
        warnings = run_rule_warnings(allow, tool_keys(tools))
        waiting = () if opts.trust_project else _untrusted_warnings(rt)
        held = (held_warning(rt.held),) if rt.global_held else ()  # no prompt here: only the app's dialog clears it
        for warning in (*rt.warnings, *held, *warnings, *waiting, *rt.commands.warnings, *rt.skills.warnings):
            sink(Notice("warn", warning))
        for note in rt.model_notices:
            sink(Notice("info", note))
        for mention in mention_notes:
            sink(Notice("info" if mention.attached else "warn", note_text(mention)))
        asyncio.run(_main(rt, opts, prompt, sink, started, state))
    except UsageError as exc:
        state.outcome = Outcome(EXIT_USAGE, "usage", str(exc))
        print(f"agent: {escape_display(str(exc))}", file=sys.stderr)
    except _Terminated:
        state.outcome = Outcome(EXIT_TERM, "interrupted")
    except KeyboardInterrupt:  # SIGINT before the loop took over: nothing was started that needs stopping
        state.outcome = Outcome(EXIT_INT, "interrupted")
    except Exception as exc:  # never a traceback: one line and exit 1
        state.outcome = Outcome(EXIT_FAIL, "error", f"internal error: {exc}")
        print(f"agent: internal error: {escape_display(str(exc))}", file=sys.stderr)
    finally:
        # From here to the end of the write a late signal is ignored: it must neither lose the result nor
        # change the exit code (the outcome's); SIGKILL is the one thing that can still cut the result short.
        saved = _ignore_signals()
        try:
            if previous is not None:
                saved[signal.SIGTERM] = previous  # what the caller had, not our launch-time handler
            if rt is not None:
                for step in (rt.abort, rt.kill_all) if not started else (rt.kill_all,):
                    with contextlib.suppress(_Terminated, KeyboardInterrupt):
                        step()  # abort: the loop never ran; kill_all: only what `close` could not stop
                # The quit check once every child process group is gone (kill_all checks too; this one is cheap).
                with contextlib.suppress(Exception, _Terminated, KeyboardInterrupt):
                    rt.config_watch.check(Cause.background())
            try:
                code = _finish(state)
            except (_Terminated, KeyboardInterrupt):
                code = state.outcome.exit_code if state.outcome else EXIT_FAIL
        finally:
            _restore_signals(saved)
    return code


def _ignore_signals() -> dict[int, object]:
    """Ignore SIGINT and SIGTERM (main thread only); what was installed before, to hand to `_restore_signals`."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {number: signal.signal(number, signal.SIG_IGN) for number in (signal.SIGINT, signal.SIGTERM)}


def _restore_signals(saved: dict[int, object]) -> None:
    for number, handler in saved.items():
        signal.signal(number, handler)  # ty: ignore[invalid-argument-type]


def _redactor(rt: Runtime) -> Callable[[str], str]:
    def redact(text: str) -> str:
        try:
            return rt.guard.redactor.redact(text)[0]
        except Exception:
            return ""  # fail closed, like the loop's hook text

    return redact


def _stdin(given: BinaryIO | None) -> tuple[BinaryIO, bool]:
    """The byte stream to read the prompt from and whether it is a terminal."""
    if given is not None:
        return given, False
    stream = sys.stdin
    if stream is None:  # closed by the caller: nothing to read
        return io.BytesIO(b""), False
    try:
        tty = stream.isatty()
    except ValueError:
        tty = False
    buffer = getattr(stream, "buffer", None)
    if buffer is None:  # a text-only replacement stream (embedding, tests): no bytes to read
        return io.BytesIO(b""), False
    return buffer, tty


def _untrusted_warnings(rt: Runtime) -> list[str]:
    """One line per kind of project setting that stays off for lack of trust (nothing prompts in headless mode)."""
    hint = "inactive: the project settings are untrusted (use --trust-project to trust them for this run)"
    lines: list[str] = []
    if (rules := rt.permissions.rules.untrusted_allow):
        lines.append(f"{len(rules)} project allow rule{'s' * (len(rules) != 1)} {hint}")
    if (entries := rt.hooks.untrusted()):
        lines.append(f"{len(entries)} project hook{'s' * (len(entries) != 1)} {hint}")
    if (servers := rt.mcp.untrusted("project")):  # the global ones held by the marker have their own warning
        lines.append(f"{len(servers)} project MCP server{'s' * (len(servers) != 1)} {hint}")
    return lines


def _parse_allow(texts: tuple[str, ...]) -> list[Rule]:
    rules: list[Rule] = []
    for text in texts:
        try:
            rules.append(parse_rule(text))
        except RuleError as exc:
            raise UsageError(f"invalid --allow rule {text!r}: {exc}") from exc
    return rules


async def _main(
    rt: Runtime, opts: HeadlessOptions, prompt: str, sink: Callable[[AgentEvent], None], started: list[bool], state: _State
) -> None:
    """`_run` as a task that SIGINT and SIGTERM cancel: its `finally` then closes the recorder and the runtime
    (every child process group), and the exit code is 128 + the signal."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(_run(rt, opts, prompt, sink, started, state))
    got: list[int] = []

    def stop(number: int) -> None:
        got.append(number)
        task.cancel()

    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stop, number)
    try:
        await task
    except asyncio.CancelledError:
        if not got:
            raise
    finally:
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)
    if got:
        state.outcome = Outcome(EXIT_INT if got[0] == signal.SIGINT else EXIT_TERM, "interrupted")


async def _run(
    rt: Runtime, opts: HeadlessOptions, prompt: str, sink: Callable[[AgentEvent], None], started: list[bool], state: _State
) -> None:
    started.append(True)
    status: list[str] = []
    errors: list[str] = []  # the error notices of the turn: the last one says why it failed

    def emit(event: AgentEvent) -> None:
        if isinstance(event, TurnEnded):
            status.append(event.status)
        elif isinstance(event, Notice) and event.level == "error":
            errors.append(event.text)
        sink(event)

    try:
        agent = rt.agent_factory(HeadlessApprover())
        try:
            if (problem := await rt.provider.check()) is not None:
                sink(Notice("error", problem))
                state.outcome = Outcome(EXIT_FAIL, "error", problem)
                return
            source = await _open_session(rt, agent, opts, sink)
            if state.collector is not None:
                state.collector.start(_session_id(agent), state.model, str(rt.root))
            await _start_mcp(rt, sink)
            await agent.session_started(source, sink)
            decision = await agent.prompt_hooks(prompt, sink)
            if decision.blocked:
                state.outcome = Outcome(EXIT_FAIL, "blocked", decision.reason)  # the hook's line is already on stderr
                return
            vision_ok, warnings = await apply_vision(
                Expansion("", tuple(state.image_notes), tuple(state.images)), rt.vision, rt.provider.model
            )
            for note in vision_ok.notes:
                sink(Notice("info", note_text(note)))
            for warning in warnings:
                sink(Notice("warn", warning))
            await agent.run_turn(prompt, emit, context=decision.context, images=vision_ok.images)
        except UsageError:
            raise
        except Exception as exc:  # an internal error is exit 1 with the message, never a traceback
            sink(Notice("error", f"internal error: {exc}"))
            state.outcome = Outcome(EXIT_FAIL, "error", f"internal error: {exc}")
            return
        finally:
            close_recorder(agent.recorder)  # flush what is queued before the id is reported
            state.session_id = _session_id(agent)
        done = status[-1:] == ["done"]
        if done:
            state.reply = _final_text(agent)
            state.outcome = Outcome(EXIT_OK, "done")
        else:
            reason: StopReason = "error"
            if status and status[-1] in ("step_limit", "interrupted"):
                reason = status[-1]
            state.outcome = Outcome(EXIT_FAIL, reason, errors[-1] if errors and reason == "error" else None)
        if not state.json_mode:
            if done:
                write_stdout(state.reply)
            if state.session_id:
                sink(Notice("info", f"session {state.session_id} saved"))
    finally:
        await rt.close()


def _session_id(agent: Agent) -> str | None:
    session_id = getattr(agent.recorder, "session_id", "")
    return session_id if isinstance(session_id, str) and session_id else None


async def _start_mcp(rt: Runtime, sink: Callable[[AgentEvent], None]) -> None:
    """Start the servers that may run and wait for them (bounded by their start timeout) before the first request."""

    def heard(event: HubEvent) -> None:
        kind, server, text = event
        match kind:
            case "ready":
                sink(Notice("info", f"mcp {server} ready · {text}"))
            case "failed" | "exited":
                sink(Notice("warn", f"mcp {server} {kind}: {text}"))
            case _:
                sink(Notice("warn", f"mcp {server}: {text}" if server else text))

    rt.mcp.listen(heard)
    rt.mcp.start()
    await rt.mcp.wait_started()


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
    old = agent.reset(
        recorder=recorder,
        history=loaded.history,
        tool_meta=loaded.tool_meta,
        todos=loaded.todos,
        usage=loaded.usage,
        unrecorded=loaded.unrecorded,
    )
    if old is not None:
        await asyncio.to_thread(close_recorder, old)
    return "resume"
