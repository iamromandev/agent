"""Running a workflow: model steps, the plan, one approval, then the calls one at a time through a child agent."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.engine.events import (
    AgentEvent,
    EventSink,
    Notice,
    TaskProgress,
    ToolProgress,
    WorkflowLine,
    WorkflowProgress,
)
from core.engine.guardrails.setup import guard_of
from core.engine.image_inputs import image_dimensions, oriented_size, resolve_input
from core.engine.imagegen import UPSCALE_SCALES, upscale_size
from core.engine.loop import Agent, Operation
from core.engine.permissions.setup import Rules
from core.engine.provider.base import Message, ProviderError, ToolCall
from core.engine.text import escape_display
from core.engine.tools.base import Tool, ToolContext, ToolResult
from core.engine.tools.paths import InvalidPath, resolve_path
from core.engine.workflows.estimates import estimate_plan, unit
from core.engine.workflows.manifest import (
    FALLBACK_ERROR,
    RETRYABLE,
    file_sha256,
    load,
    model_steps_digest,
    new_manifest,
    save,
    shape_ok,
)
from core.engine.workflows.model import PlannedCall, PlanSummary, RunResult, Workflow
from core.engine.workflows.plan import PlanError, digest, input_refs, plan_calls
from core.engine.workflows.retry import Merge, feedback, merge_calls, model_deps, redo_targets
from core.engine.workflows.runs import RunEntry, RunIndex
from core.engine.workflows.schema import validate
from core.engine.workflows.template import Scope, TemplateError, render, render_text
from core.engine.workflows.timings import TimingStore

Approve = Callable[[PlanSummary], Awaitable[bool]]
SLUG_CHARS = 24
REPLY_SHOWN = 200
ERROR_SHOWN = 200
ERROR_SAVED = 300
RUN_ATTEMPTS = 100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SYSTEM = (
    "Answer with only one JSON value that matches this JSON Schema. No other text and no code fence.\n"
    "JSON Schema:\n{schema}"
)
RETRY = "That answer is not valid: {error}\nAnswer again with only one JSON value that matches the schema."
NOT_JSON = "the answer is not JSON"
CHANGED_BY_HAND = "the manifest was changed by hand; start a new run"
FENCE = re.compile(r"```[A-Za-z]*\s*\n?(.*?)```", re.DOTALL)
NON_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """At most 24 characters of `[a-z0-9-]` made from the input; `run` when nothing is left."""
    return NON_SLUG.sub("-", text.lower()).strip("-")[:SLUG_CHARS].strip("-") or "run"


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


DECODER = json.JSONDecoder(parse_constant=_no_constant)  # NaN, Infinity and -Infinity are not standard JSON
MAX_STARTS = 200  # candidate `{`/`[` positions tried in a reply, so a reply of 200,000 brackets costs little


MAX_JSON_DEPTH = 64  # a reply nested deeper is no answer (and could trip recursion later in templates or digests)


def _too_deep(value: Any) -> bool:
    todo = [(value, 1)]
    while todo:
        item, depth = todo.pop()
        if isinstance(item, list | dict):
            if depth > MAX_JSON_DEPTH:
                return True
            todo.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return False


def _loads(text: str) -> tuple[bool, Any]:
    try:
        value = DECODER.decode(text)
    except (ValueError, RecursionError):
        return False, None
    return (False, None) if _too_deep(value) else (True, value)


def first_json(text: str) -> tuple[bool, Any]:
    """The first JSON value in a model's reply: the whole text, else a fenced block, else the first value that
    decodes from a `{` or `[`. `(False, None)` when there is none."""
    found, value = _loads(text.strip())
    if found:
        return True, value
    for block in FENCE.findall(text):
        found, value = _loads(block.strip())
        if found:
            return True, value
    tried = 0
    for start, char in enumerate(text):
        if char in "{[":
            tried += 1
            if tried > MAX_STARTS:
                break
            try:
                value = DECODER.raw_decode(text, start)[0]
            except (ValueError, RecursionError):
                continue
            if not _too_deep(value):
                return True, value
    return False, None


class _Failed(Exception):
    """A run that stops before or between calls; carries the line to show."""

    def __init__(self, line: str) -> None:
        super().__init__(line)
        self.line = line


@dataclass
class _Loaded:
    """A saved run that passed the checks every resume makes."""

    wf: Workflow
    run: str
    rel: str  # the run folder, relative to the workspace
    folder: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _Retry:
    targets: list[str]  # model steps to ask again, in workflow order
    tool: str  # the call that was refused ("" for a redo the user asked for)
    key: str
    error: str | None  # None: the user asked for a new answer
    direct: frozenset[str]  # the targets asked because of the refusal (or named by the user); the rest depend on them


class WorkflowRunner:
    def __init__(
        self,
        agent: Agent,
        tools: Callable[[], Sequence[Tool]],
        ctx: ToolContext,
        rules: Rules,
        index: RunIndex,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        timings: TimingStore | None = None,
    ) -> None:
        self._agent = agent
        self._tools = tools
        self._ctx = ctx
        self._rules = rules
        self._index = index
        self._now = now
        self._timings = timings if timings is not None else TimingStore(None)  # None: nothing is written
        self.saved = False  # the current or last run has written its manifest: a cancel can be resumed

    # -- text safety ---------------------------------------------------------------------------

    def _text(self, text: str) -> str:
        """Model or tool text that is shown: redacted (nothing of it when redaction fails), control characters out."""
        try:
            return escape_display(guard_of(self._ctx).redactor.redact(text)[0])
        except Exception:
            return ""

    # -- the run folder ------------------------------------------------------------------------

    def _folder_error(self, rel: str) -> tuple[Path | None, str]:
        parts = rel.split("/")
        if not rel or rel.startswith(("/", "~")) or ".." in parts or any(p.casefold() in (".agent", ".git") for p in parts):
            return None, f"the run folder {escape_display(rel)} is not a plain relative path"
        try:
            path = resolve_path(self._ctx, rel)
        except InvalidPath as exc:
            return None, f"the run folder {escape_display(rel)} is not valid: {escape_display(str(exc))}"
        root = self._ctx.workspace.resolve()
        if path == root or not path.is_relative_to(root):
            return None, f"the run folder {escape_display(rel)} is outside the workspace"
        if guard_of(self._ctx).paths.is_protected(path):
            return None, f"the run folder {escape_display(rel)} is a protected path"
        return path, ""

    def _new_run(self, wf: Workflow, text: str) -> tuple[str, str, Path]:
        """`(run id, relative folder, folder)`: the first id whose folder does not exist yet. Raises `_Failed`."""
        stamp = self._now().astimezone(UTC).strftime("%Y%m%d-%H%M")
        base = f"{stamp}-{slug(text)}"
        error = f"workflow {wf.name} failed at folder: no free run folder"
        for n in range(1, RUN_ATTEMPTS + 1):
            run = base if n == 1 else f"{base}-{n}"
            try:
                rel = render(wf.folder, Scope({"name": wf.name, "run": run, "input": text}))
            except TemplateError as exc:
                raise _Failed(f"workflow {wf.name} failed at folder: {self._text(str(exc))}") from exc
            folder, problem = self._folder_error(str(rel))
            if folder is None:
                raise _Failed(f"workflow {wf.name} failed at folder: {problem}")
            if not folder.exists():
                return run, str(rel), folder
        raise _Failed(error)

    # -- model steps ---------------------------------------------------------------------------

    async def _model_steps(self, wf: Workflow, scope: Scope, emit: EventSink) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for step in wf.steps:
            if step.kind != "model":
                continue
            emit(WorkflowProgress(f"workflow {escape_display(wf.name)} · asking the model · {escape_display(step.id)}"))
            assert step.schema is not None
            try:
                prompt = render_text(step.prompt, Scope({**scope.values, **outputs}))
            except TemplateError as exc:
                raise _Failed(f"workflow {wf.name} failed at {step.id}: {self._text(str(exc))}") from exc
            schema = json.dumps(step.schema, ensure_ascii=False, sort_keys=True)
            messages = [Message("system", SYSTEM.format(schema=schema)), Message("user", prompt)]
            outputs[step.id] = await self._ask(wf, step.id, step.schema, messages)
        return outputs

    async def _ask(self, wf: Workflow, step_id: str, schema: dict[str, Any], messages: list[Message]) -> Any:
        reply = ""
        error = ""
        for attempt in (1, 2):
            answer = await self._agent.ask_json(messages)
            if isinstance(answer, ProviderError):
                raise _Failed(f"workflow {wf.name} failed at {step_id}: {self._text(answer.message)}")
            reply = answer
            found, value = first_json(reply)
            problem = validate(schema, value) if found else NOT_JSON
            if problem is None:
                return value
            error = problem
            if attempt == 1:
                messages = [*messages, Message("assistant", reply), Message("user", RETRY.format(error=problem))]
        shown = self._text(reply[:REPLY_SHOWN])
        raise _Failed(f"workflow {wf.name} failed at {step_id}: {self._text(error)}; the reply began: {shown}")

    # -- planning ------------------------------------------------------------------------------

    def _plan(self, wf: Workflow, scope: Scope, folder: Path) -> list[PlannedCall]:
        tools: Mapping[str, Tool] = {t.name: t for t in self._tools()}
        try:
            return plan_calls(wf, scope, tools, self._ctx, self._rules, folder)
        except PlanError as exc:
            raise _Failed(f"workflow {wf.name} failed at plan: {self._text(str(exc))}") from exc

    @staticmethod
    def _scale(args: dict[str, Any]) -> int:
        scale = args.get("scale")
        return scale if type(scale) is int and scale in UPSCALE_SCALES else 2

    def _source_pixels(self, args: dict[str, Any]) -> int | None:
        """Output pixels of an `upscale_image` call, from its source file's size (None when it cannot be read)."""
        ref = args.get("image")
        if not isinstance(ref, str) or ref.startswith("clipboard:"):
            return None
        try:
            item = resolve_input(ref, self._ctx)
        except Exception:
            return None
        if isinstance(item, ToolResult):
            return None
        width, height = oriented_size(item)
        try:
            out_w, out_h = upscale_size(width, height, self._scale(args))
        except ArithmeticError:
            return None
        return out_w * out_h

    def _output_pixels(self, args: dict[str, Any]) -> int | None:
        """Pixels of the PNG a call wrote at its `path`, from the header (None when it is not a readable PNG)."""
        raw = args.get("path")
        if not isinstance(raw, str):
            return None
        try:
            fd = os.open(resolve_path(self._ctx, raw), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except (OSError, InvalidPath):
            return None
        try:
            head = os.read(fd, 32)
        except OSError:
            return None
        finally:
            os.close(fd)
        size = image_dimensions("png", head) if head.startswith(PNG_SIGNATURE) else None
        return size[0] * size[1] if size else None

    def _summary(
        self,
        wf: Workflow,
        rel: str,
        run: str,
        calls: Sequence[PlannedCall],
        notes: tuple[str, ...] = (),
        reused: int = 0,
    ) -> PlanSummary:
        counts: dict[str, int] = {}
        for call in calls:
            counts[call.tool] = counts.get(call.tool, 0) + 1
        seconds, source = estimate_plan(calls, self._timings, self._source_pixels)
        return PlanSummary(wf.name, rel, run, tuple(calls), counts, seconds, source, notes, reused)

    # -- running -------------------------------------------------------------------------------

    async def run(self, wf: Workflow, text: str, emit: EventSink, approve: Approve) -> RunResult:
        """Model steps, plan, approval, then the calls. One undo unit; raises `AgentBusy` when a turn is running and
        lets a cancel through (after saving the manifest). The result's `line` is the outcome to show."""
        prompt = f"/workflow run {wf.name} {text}".rstrip()
        self.saved = False
        run, rel = "", ""
        try:
            async with self._agent.operation(prompt, emit) as op:
                op.keep = False  # nothing changes until the plan is approved: a failure before leaves no undo unit
                run, rel, folder = self._new_run(wf, text)
                scope = Scope({"name": wf.name, "folder": rel, "run": run, "input": text})
                outputs = await self._model_steps(wf, scope, emit)
                calls = self._plan(wf, Scope({**scope.values, **outputs}), folder)
                if not await approve(self._summary(wf, rel, run, calls)):
                    return RunResult(False, run, rel, f"workflow {wf.name}: the plan was not approved")
                manifest = new_manifest(wf, run, rel, text, outputs, calls)
                return await self._execute(wf, run, rel, folder, manifest, calls, list(range(len(calls))), emit, op)
        except _Failed as failed:
            return RunResult(False, run, rel, failed.line)

    async def resume(
        self,
        wf_lookup: Callable[[str], Workflow | None],
        run_id: str | None,
        emit: EventSink,
        approve: Approve,
        redo: str | None = None,
    ) -> RunResult:
        """Run what is left of a run (the latest unfinished one, or `run_id`): the saved model answers are reused and
        only calls that are not done, or whose output file changed, run. After an `invalid` refusal, or with `redo`
        (a model step id), the model steps behind the call are asked again first and the run is planned again. A new
        undo unit."""
        self.saved = False
        try:
            loaded = self._load_run(wf_lookup, run_id, redo is not None)
            retry = self._retry(loaded, redo)
        except _Failed as failed:
            return RunResult(False, run_id or "", "", failed.line)
        if retry is not None:
            return await self._redo(loaded, retry, emit, approve)
        try:
            wf, run, rel, folder, manifest = loaded.wf, loaded.run, loaded.rel, loaded.folder, loaded.manifest
            calls, pending = self._pending_plain(loaded)
        except _Failed as failed:
            return RunResult(False, run_id or "", "", failed.line)
        if not pending:
            self._index.record(run, wf.name, rel, "done")
            return RunResult(True, run, rel, "nothing to resume")
        try:
            async with self._agent.operation(f"/workflow resume {run}", emit) as op:
                op.keep = False
                if not await approve(self._summary(wf, rel, run, [calls[p] for p in pending])):
                    return RunResult(False, run, rel, f"workflow {wf.name}: the plan was not approved")
                return await self._execute(wf, run, rel, folder, manifest, calls, pending, emit, op)
        except _Failed as failed:
            return RunResult(False, run, rel, failed.line)

    def _has_manifest(self, entry: RunEntry) -> bool:
        folder, _ = self._folder_error(entry.folder)
        return folder is not None and load(folder) is not None

    def _retry(self, loaded: _Loaded, redo: str | None) -> _Retry | None:
        """What to ask the model again, or None for the plain resume. Raises `_Failed` for a bad `redo`."""
        wf = loaded.wf
        if redo is not None:
            models = [s.id for s in wf.steps if s.kind == "model"]
            if redo not in models:
                raise _Failed(
                    f"no model step '{escape_display(redo)}' in workflow {wf.name} (model steps: {', '.join(models)})"
                )
            return _Retry(redo_targets(wf, {redo}), "", "", None, frozenset({redo}))
        failed = [c for c in loaded.manifest["calls"] if c["state"] == "failed"]
        if not failed or failed[-1]["summary"] != RETRYABLE:
            return None
        last = failed[-1]
        deps = model_deps(wf, last["key"].split("/")[0])
        if not deps:
            return None
        error = (last.get("error") or FALLBACK_ERROR)[:ERROR_SAVED]  # the file may have been edited: bound what is used
        return _Retry(redo_targets(wf, deps), last["tool"][:ERROR_SAVED], last["key"][:ERROR_SAVED], error, frozenset(deps))

    def _redacted(self, text: str) -> str:
        """Tool text on its way to the model: redacted, and the fallback text when redaction fails."""
        try:
            return guard_of(self._ctx).redactor.redact(text)[0]
        except Exception:
            return FALLBACK_ERROR

    async def _redo(self, loaded: _Loaded, retry: _Retry, emit: EventSink, approve: Approve) -> RunResult:
        """Ask the model steps again, plan again and run what changed. Everything happens on a copy of the manifest:
        the file is replaced (once, atomically) only after the approval, so any stop before leaves it as it was."""
        wf, run, rel, folder = loaded.wf, loaded.run, loaded.rel, loaded.folder
        try:
            typed = f"/workflow resume {run}" + (f" --redo {retry.targets[0]}" if retry.error is None else "")
            async with self._agent.operation(typed, emit) as op:
                op.keep = False
                manifest = copy.deepcopy(loaded.manifest)
                outputs = manifest["outputs"]
                base = {"name": wf.name, "folder": rel, "run": run, "input": manifest["input"]}
                for step_id in retry.targets:
                    step = next(s for s in wf.steps if s.id == step_id)
                    assert step.schema is not None
                    emit(WorkflowProgress(f"workflow {escape_display(wf.name)} · asking the model · {escape_display(step_id)} (retry)"))
                    try:
                        prompt = render_text(step.prompt, Scope({**base, **outputs}))
                    except TemplateError as exc:
                        raise _Failed(f"workflow {wf.name} failed at {step_id}: {self._text(str(exc))}") from exc
                    error = None if retry.error is None else self._redacted(retry.error)
                    tool, key = self._redacted(retry.tool), self._redacted(retry.key)
                    messages = [
                        Message("system", SYSTEM.format(schema=json.dumps(step.schema, ensure_ascii=False, sort_keys=True))),
                        Message("user", prompt),
                        Message("assistant", json.dumps(outputs[step_id], ensure_ascii=False)),
                        Message("user", feedback(tool, key, error, dependent=step_id not in retry.direct)),
                    ]
                    outputs[step_id] = await self._ask(wf, step_id, step.schema, messages)
                saved = loaded.manifest["calls"]
                calls = self._plan(wf, Scope({**base, **outputs}), folder)
                merge = self._merge(calls, saved)
                changed = self._tool_steps_changed(wf, base, loaded.manifest, folder)
                dropped = len({e["key"] for e in saved} - {c.key for c in calls})
                if not merge.pending and not dropped and not changed:
                    return RunResult(True, run, rel, "nothing changed; nothing to run")
                why = "asked by the user" if retry.error is None else self._text(retry.error)
                notes = [f"re-asked {escape_display(step)}: {why}" for step in retry.targets]
                if changed:
                    notes.append("tool steps changed since this run")
                if dropped:
                    notes.append(f"{dropped} saved call{'' if dropped == 1 else 's'} dropped")
                # With nothing pending but entries dropped or changed the dialog still opens (0 calls and the notes):
                # the cleaned manifest is written only after the approval.
                summary = self._summary(wf, rel, run, [calls[p] for p in merge.pending], tuple(notes), merge.reused)
                if not await approve(summary):
                    return RunResult(False, run, rel, f"workflow {wf.name}: the plan was not approved")
                manifest["calls"] = merge.entries  # saved by `_execute`, once, before the first call
                return await self._execute(wf, run, rel, folder, manifest, calls, merge.pending, emit, op)
        except _Failed as failed:
            return RunResult(False, run, rel, failed.line)

    def _tool_steps_changed(
        self, wf: Workflow, base: dict[str, Any], manifest: dict[str, Any], folder: Path
    ) -> bool:
        """Whether the workflow's tool steps, planned from the answers the run saved, no longer give the saved calls
        (the file was edited after the run; the model steps are unchanged, `_load_run` checked that)."""
        outputs = manifest["outputs"]
        answers = {s.id: outputs[s.id] for s in wf.steps if s.kind == "model"}
        try:
            old = self._plan(wf, Scope({**base, **answers}), folder)
        except _Failed:
            return True
        return [(c.key, c.digest) for c in old] != [(c["key"], c["digest"]) for c in manifest["calls"]]

    def _merge(self, calls: Sequence[PlannedCall], saved: Sequence[dict[str, Any]]) -> Merge:
        def unchanged(call: PlannedCall, entry: dict[str, Any]) -> bool:
            path, stored = call.args.get("path"), entry["output_sha256"]
            return stored is None or not isinstance(path, str) or file_sha256(resolve_path(self._ctx, path)) == stored

        return merge_calls(
            calls,
            saved,
            unchanged,
            lambda call, rewritten: bool(self._resolved(input_refs(call.args)) & rewritten),
            lambda call: self._resolved([call.args.get("path")]),
        )

    def _pending_plain(self, loaded: _Loaded) -> tuple[list[PlannedCall], list[int]]:
        """Plan from the saved answers (the plan must be the saved one) and find the calls that still have to run."""
        wf, manifest = loaded.wf, loaded.manifest
        saved = manifest["calls"]
        outputs = manifest["outputs"]
        answers = {s.id: outputs[s.id] for s in wf.steps if s.kind == "model"}  # only the model steps' names
        scope = Scope({"name": wf.name, "folder": loaded.rel, "run": loaded.run, "input": manifest["input"], **answers})
        calls = self._plan(wf, scope, loaded.folder)
        if [(c.key, c.digest) for c in calls] != [(c["key"], c["digest"]) for c in saved]:
            raise _Failed("the workflow's tool steps changed; start a new run")
        merge = self._merge(calls, saved)
        manifest["calls"] = merge.entries
        return calls, merge.pending

    def _load_run(
        self, wf_lookup: Callable[[str], Workflow | None], run_id: str | None, any_state: bool = False
    ) -> _Loaded:
        """`any_state`: with no `run_id` and no unfinished run, the latest run of any state (a finished one can be
        redone)."""
        if run_id:
            entry = self._index.find(run_id)
        else:
            entry = self._index.latest_unfinished(self._has_manifest)
            if entry is None and any_state:
                entry = self._index.latest(self._has_manifest)
        if entry is None:
            raise _Failed(f"no workflow run '{escape_display(run_id)}'" if run_id else "no unfinished workflow run to resume")
        folder, problem = self._folder_error(entry.folder)
        if folder is None:
            raise _Failed(problem)
        manifest = load(folder)
        if manifest is None:
            raise _Failed(f"the manifest of run {escape_display(entry.run)} is missing or unreadable")
        if not shape_ok(manifest):
            raise _Failed(CHANGED_BY_HAND)
        name = manifest["workflow"]
        wf = wf_lookup(name)
        if wf is None:
            raise _Failed(f"workflow {escape_display(name)} is not available")
        if model_steps_digest(wf) != manifest.get("model_steps_sha256"):
            raise _Failed("the workflow's model steps changed; start a new run")
        saved = manifest["calls"]
        try:
            digests_ok = all(digest(c["args"]) == c["digest"] for c in saved)
        except ValueError:  # an argument that is not JSON (NaN)
            digests_ok = False
        outputs = manifest["outputs"]
        if not digests_ok or any(s.id not in outputs for s in wf.steps if s.kind == "model"):
            raise _Failed(CHANGED_BY_HAND)
        return _Loaded(wf, entry.run, entry.folder, folder, manifest)

    def _resolved(self, refs: Sequence[Any]) -> set[Path]:
        """The resolved paths of the string references (anything that does not resolve is left out)."""
        found: set[Path] = set()
        for ref in refs:
            if isinstance(ref, str) and not ref.startswith("clipboard:"):
                try:
                    found.add(resolve_path(self._ctx, ref))
                except InvalidPath:
                    continue
        return found

    def _child(self, run: str, calls: Sequence[PlannedCall]) -> Agent:
        """The agent that runs the calls: the main agent's rules, grants, hooks, guard and undo log, its own history.
        It skips the approval dialog for exactly the calls of the approved plan, nothing else."""
        agent = self._agent
        planned = {f"wf:{run}:{c.key}" for c in calls}
        child = Agent(
            agent.provider,
            list(self._tools()),
            agent.approver,
            self._ctx,
            agent.max_steps,
            today=agent.today,
            recorder=None,
            role="task",
            vision=agent.vision,
            rules=self._rules,
            watch=agent.watch,
            hooks=agent.hooks,
            subagent="workflow",
            mode_source=lambda: agent.mode,
            undo=agent.undo_log,
            preapproved=lambda call: call.id in planned,
        )
        child.grants = agent.grants
        return child

    def _sink(self, emit: EventSink, label: str) -> Callable[[AgentEvent], None]:
        def sink(event: AgentEvent) -> None:
            match event:
                case TaskProgress(_, ToolProgress(text)):
                    emit(WorkflowProgress(f"{label} · {self._text(text)}"))
                case Notice(level, text):
                    emit(WorkflowLine("info" if level == "info" else "warn", self._text(text)))

        return sink

    def _save(self, folder: Path, manifest: dict[str, Any], run: str, name: str, rel: str, state: str) -> None:
        """Write the manifest and the index. A failure is a `_Failed` with the line to show."""
        try:
            save(folder, manifest)
            self._index.record(run, name, rel, state)
        except (OSError, ValueError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
            raise _Failed(f"workflow {name} failed: could not save the manifest: {self._text(reason)}") from exc

    def _save_quiet(self, folder: Path, manifest: dict[str, Any], run: str, name: str, rel: str, state: str) -> None:
        """`_save` for a cancel: whatever goes wrong, the cancel is what gets through."""
        with contextlib.suppress(Exception):
            self._save(folder, manifest, run, name, rel, state)

    async def _execute(
        self,
        wf: Workflow,
        run: str,
        rel: str,
        folder: Path,
        manifest: dict[str, Any],
        calls: Sequence[PlannedCall],
        pending: list[int],
        emit: EventSink,
        op: Operation,
    ) -> RunResult:
        name = wf.name
        child = self._child(run, calls)
        per_step: dict[str, int] = {}
        for call in calls:
            per_step[call.step] = per_step.get(call.step, 0) + 1
        saved = manifest["calls"]
        # What the manifest says is done is what the plan says now; `saved` entries are updated as each call ends.
        self._save(folder, manifest, run, name, rel, "running")
        # Only now, and only when a call will run: a first save that failed changed nothing, so it leaves no undo unit.
        op.keep = bool(pending)
        self.saved = True
        timing_failed = False  # one warning per run, then no more writes
        for pos in pending:
            call, entry = calls[pos], saved[pos]
            label = f"workflow {escape_display(name)} · {escape_display(call.step)} {call.index or 1}/{per_step[call.step]}"
            emit(WorkflowProgress(label))
            tool_call = ToolCall(f"wf:{run}:{call.key}", call.tool, dict(call.args))
            started = time.monotonic()
            try:
                result = await child.run_call(tool_call, self._sink(emit, label))
            except asyncio.CancelledError:
                entry.update(state="failed", summary="interrupted")
                self._save_quiet(folder, manifest, run, name, rel, "failed")
                raise
            except Exception as exc:
                result = ToolResult(f"{type(exc).__name__}: {exc}", True, "failed")
            raw_path = call.args.get("path")
            shown = self._text(raw_path) if isinstance(raw_path, str) else escape_display(call.step)
            if result.is_error:
                first = (result.output.strip().splitlines() or [""])[0]
                entry.update(state="failed", summary=result.summary, error=first[:ERROR_SAVED])
                self._save(folder, manifest, run, name, rel, "failed")
                reason = self._text(first)[:ERROR_SHOWN]
                emit(WorkflowLine("warn", f"✗ {escape_display(call.tool)} {shown} ({reason})"))
                where = f"{escape_display(call.step)} {call.index or 1}/{per_step[call.step]}"
                again = result.summary == RETRYABLE and bool(model_deps(wf, call.step))
                ending = "/workflow resume asks the model again" if again else "/workflow resume to continue"
                line = f"workflow {name} failed at {where}: {reason}; {ending}"
                return RunResult(False, run, rel, line)
            output = file_sha256(resolve_path(self._ctx, raw_path)) if isinstance(raw_path, str) else None
            entry.pop("error", None)
            entry.update(state="done", summary=result.summary, output_sha256=output)
            self._save(folder, manifest, run, name, rel, "running")
            if not timing_failed:
                try:  # a timing record must never stop a run after a call that worked
                    key, amount = unit(call.tool, call.args, self._output_pixels)
                    self._timings.add(key, (time.monotonic() - started) / amount)
                except Exception as exc:
                    timing_failed = True
                    reason = (exc.strerror if isinstance(exc, OSError) else None) or str(exc)
                    emit(WorkflowLine("warn", f"could not save the call timings: {self._text(reason)}"))
            emit(WorkflowLine("info", f"✓ {escape_display(call.tool)} {shown} ({self._text(result.summary)})"))
        self._save(folder, manifest, run, name, rel, "done")
        return RunResult(True, run, rel, f"workflow {name} finished: {len(pending)} calls, files in {escape_display(rel)}")
