"""`run_headless`: one agent turn without Textual; the final reply on stdout, progress on stderr, an exit code."""

from __future__ import annotations

import asyncio
import io
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Literal

from src.engine import Agent
from src.engine.commands.builtins import BUILTINS
from src.engine.compaction import STOP_NOTE_PREFIX
from src.engine.events import AgentEvent, Notice, TurnEnded
from src.engine.loop import close_recorder
from src.engine.mcp.contract import McpConnector
from src.engine.mcp.hub import HubEvent
from src.engine.permissions.rules import Rule, RuleError, parse_rule
from src.engine.permissions.setup import run_rule_warnings, tool_keys
from src.engine.provider.base import SwitchableProvider
from src.engine.session import FileRecorder, SessionError, SessionWriteError
from src.engine.text import escape_display
from src.engine.tools import default_tools
from src.headless import EXIT_FAIL, EXIT_INT, EXIT_OK, EXIT_TERM, EXIT_USAGE, HeadlessOptions, UsageError
from src.headless.approver import HeadlessApprover
from src.headless.output import StderrSink, want_color, write_stdout
from src.headless.prompt import build_prompt
from src.runtime import Runtime, build_runtime


class _Terminated(BaseException):
    """SIGTERM before the loop took over (`_main` then handles the signal itself)."""


def _raise_terminated(number: int, frame: object) -> None:
    raise _Terminated


def run_headless(
    workspace: Path,
    opts: HeadlessOptions,
    *,
    provider: SwitchableProvider | None = None,
    connector: McpConnector | None = None,
    stdin: BinaryIO | None = None,
) -> int:
    """Run one turn for `opts` in `workspace` and return the exit code. `provider` replaces Ollama, `connector` the
    MCP servers' and `stdin` the process's (bytes; all for tests)."""
    sink = StderrSink(sys.stderr, opts.quiet, want_color(sys.stderr))
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
        )
        prompt = build_prompt(opts.prompt, *_stdin(stdin), rt.commands, BUILTINS)
        rt.permissions.rules.add_run_allow(allow)
        if opts.trust_project:
            rt.permissions.rules.trust_for_run()  # hooks follow the rules' trust; no store is written
        tools = default_tools(None, fetcher=rt.web, searcher=rt.search, skills=rt.skills)
        warnings = run_rule_warnings(allow, tool_keys(tools))
        waiting = () if opts.trust_project else _untrusted_warnings(rt)
        for warning in (*rt.warnings, *warnings, *waiting, *rt.commands.warnings, *rt.skills.warnings):
            sink(Notice("warn", warning))
        for note in (*rt.model_notices, *rt.guard.info):
            sink(Notice("info", note))
        return asyncio.run(_main(rt, opts, prompt, sink, started))
    except UsageError as exc:
        print(f"agent: {escape_display(str(exc))}", file=sys.stderr)
        return EXIT_USAGE
    except _Terminated:
        return EXIT_TERM
    except KeyboardInterrupt:  # SIGINT before the loop took over: nothing was started that needs stopping
        return EXIT_INT
    except Exception as exc:  # never a traceback: one line and exit 1
        print(f"agent: internal error: {escape_display(str(exc))}", file=sys.stderr)
        return EXIT_FAIL
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        if rt is not None:
            if not started:
                rt.abort()  # the loop never ran, so nothing closed the clients
            rt.kill_all()  # after `close` ran in the loop, this only covers what it could not stop


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
    if (servers := rt.mcp.untrusted()):
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
    rt: Runtime, opts: HeadlessOptions, prompt: str, sink: Callable[[AgentEvent], None], started: list[bool]
) -> int:
    """`_run` as a task that SIGINT and SIGTERM cancel: its `finally` then closes the recorder and the runtime
    (every child process group), and the exit code is 128 + the signal."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(_run(rt, opts, prompt, sink, started))
    got: list[int] = []

    def stop(number: int) -> None:
        got.append(number)
        task.cancel()

    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stop, number)
    try:
        code = await task
    except asyncio.CancelledError:
        if not got:
            raise
        code = EXIT_FAIL
    finally:
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)
    return (EXIT_INT if got[0] == signal.SIGINT else EXIT_TERM) if got else code


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
            await _start_mcp(rt, sink)
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
    old = agent.reset(recorder=recorder, history=loaded.history, tool_meta=loaded.tool_meta, todos=loaded.todos)
    if old is not None:
        await asyncio.to_thread(close_recorder, old)
    return "resume"
