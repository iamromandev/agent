"""The `Hooks` a run holds: which entries are active, running them for an event, and reading what they answer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.engine.events import Notice
from src.engine.hooks.config import TOOL_EVENTS, Event, HookEntry, read_hooks
from src.engine.hooks.runner import REASON_CHARS, HookRun, Runner
from src.engine.permissions.setup import PermissionSetup
from src.engine.text import escape

Outcome = Literal["ok", "blocked", "context", "warning", "timeout"]

BLOCKING_EVENTS: frozenset[str] = frozenset({"PreToolUse", "UserPromptSubmit", "Stop"})
DEFAULT_REASON = "blocked by hook"
NOTICE_CHARS = 200  # how much of a hook's text one transcript line shows


def notice_text(text: str) -> str:
    """Hook text for one transcript line: cut first, then escaped, so the cut never splits an escape."""
    return escape(text[:NOTICE_CHARS])


class _NullRunner:
    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: Any) -> HookRun:
        raise AssertionError("an empty Hooks has nothing to run")


@dataclass(frozen=True)
class HookOutcome:
    entry: HookEntry
    outcome: Outcome
    text: str  # reason, context or warning as the hook gave it: raw (escape to show, redact for the model)


@dataclass(frozen=True)
class HookDecision:
    blocked: bool
    reason: str  # the first block's reason; DEFAULT_REASON when it gave none
    context: list[str]  # in run order
    outcomes: list[HookOutcome]


class Hooks:
    def __init__(
        self,
        entries: Sequence[HookEntry],
        warnings: Sequence[str],
        runner: Runner,
        project_active: Callable[[], bool],
        workspace: Path,
    ) -> None:
        self.entries = tuple(entries)
        self.warnings = tuple(warnings)
        self._runner = runner
        self._project_active = project_active
        self._workspace = workspace

    @classmethod
    def empty(cls) -> Hooks:
        return cls((), (), _NullRunner(), lambda: False, Path())

    def _active(self, event: Event, tool: str | None) -> list[HookEntry]:
        if not self.entries:
            return []
        project = self._project_active()
        chosen = [
            e
            for e in self.entries
            if e.event == event and (e.source == "global" or project) and (tool is None or e.matches(tool))
        ]
        return sorted(chosen, key=lambda e: e.source != "global")  # stable: global first, each in file order

    def has(self, event: Event, tool: str | None = None) -> bool:
        """Is there an active entry for this event (and tool)? Cheap: no payload is built before it says yes."""
        return bool(self._active(event, tool))

    def untrusted(self) -> tuple[HookEntry, ...]:
        """The project entries while they wait for the user's trust."""
        if self._project_active():
            return ()
        return tuple(e for e in self.entries if e.source == "project")

    async def run(self, event: Event, payload: dict[str, Any], tool: str | None = None) -> HookDecision:
        """Run the active matching entries one after another. Never raises (but for a cancel)."""
        stdin = json.dumps({"event": event, **payload}, default=str).encode()
        env = {"AGENT_HOOK_EVENT": event, "AGENT_PROJECT_DIR": str(self._workspace)}
        outcomes: list[HookOutcome] = []
        context: list[str] = []
        blocked = False
        reason = DEFAULT_REASON
        for entry in self._active(event, tool):
            try:
                run = await self._runner.run(entry.command, stdin, entry.timeout, env)
                outcome = _interpret(entry, run, event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = HookOutcome(entry, "warning", f"hook failed: {exc}")
            outcomes.append(outcome)
            if outcome.outcome == "context":
                context.append(outcome.text)
            elif outcome.outcome == "blocked":
                blocked, reason = True, outcome.text or DEFAULT_REASON
                break  # R9: the first block ends the event's hooks
        return HookDecision(blocked, reason, context, outcomes)

    def listing(self) -> list[str]:
        """The lines `/hooks` shows: every entry as `<Event> [<matcher>] (<source>, <timeout>[, inactive: untrusted]) <command>`,
        then the launch warnings; empty when there is nothing to show. Text from a file is escaped."""
        waiting = set(self.untrusted())
        lines = [
            f"{escape(e.label)} ({e.source}, {e.timeout:g}s{', inactive: untrusted' if e in waiting else ''}) {escape(e.command)}"
            for e in self.entries
        ]
        return [*lines, *self.warnings]

    def notices(self, decision: HookDecision, redact: Callable[[str], str] | None = None) -> list[Notice]:
        """One transcript line for every hook that blocked, added context, warned or timed out. `redact` (the guard's)
        takes secrets out of a hook's text first: it is cut and escaped only after that."""
        notices: list[Notice] = []
        for o in decision.outcomes:
            if o.outcome == "ok":
                continue
            text = notice_text(redact(o.text) if redact is not None else o.text)
            line = f"hook {o.entry.event} ({o.entry.source}) {o.outcome}" + (f": {text}" if text else "")
            notices.append(Notice("warn" if o.outcome in ("warning", "timeout") else "info", line))
        return notices


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _interpret(entry: HookEntry, run: HookRun, event: Event) -> HookOutcome:
    """What one run says. A block on an event that cannot block becomes context (`PostToolUse`) or a warning
    (`SessionStart`); `PreToolUse` context is a warning (nothing uses it)."""
    if run.error is not None:
        return HookOutcome(entry, "warning", run.error)
    if run.timed_out:
        return HookOutcome(entry, "timeout", f"timed out after {entry.timeout:g}s")
    if run.exit_code == 2:
        return _block(entry, run.stderr.strip()[:REASON_CHARS], event)
    if run.exit_code != 0:
        return HookOutcome(entry, "warning", f"exit {run.exit_code}: {run.stderr.strip()[:REASON_CHARS]}".rstrip(": "))
    answer = _json_object(run.stdout)
    context = _string(answer.get("context"))
    if answer.get("decision") == "block":
        return _block(entry, _string(answer.get("reason"))[:REASON_CHARS], event, context)
    if context and event == "PreToolUse":  # a tool call has no place for it: say so instead of dropping it silently
        return HookOutcome(entry, "warning", "PreToolUse context is unused; ignored")
    return HookOutcome(entry, "context", context) if context else HookOutcome(entry, "ok", "")


def _block(entry: HookEntry, reason: str, event: Event, context: str = "") -> HookOutcome:
    if event in BLOCKING_EVENTS:
        return HookOutcome(entry, "blocked", reason)
    if event in TOOL_EVENTS:  # PostToolUse cannot block: what it says goes to the model as context
        text = "\n".join(part for part in (reason, context) if part)
        return HookOutcome(entry, "context", text) if text else HookOutcome(entry, "ok", "")
    return HookOutcome(entry, "warning", f"{event} hooks cannot block; ignored")


def prepare(setup: PermissionSetup, workspace: Path, runner: Runner) -> Hooks:
    """The launch's `Hooks` from both settings files' parsed data. Project entries run only while the project file is
    trusted (the same decision as its allow rules)."""
    global_entries, global_warnings = read_hooks(setup.global_data, "global")
    project_entries, project_warnings = read_hooks(setup.project_data, "project")
    rules = setup.rules
    return Hooks(
        (*global_entries, *project_entries),
        (*global_warnings, *project_warnings),
        runner,
        lambda: rules.project_trusted,
        workspace,
    )
