"""Retrying a run after a tool refused the model's arguments: which model steps to ask again, what to tell the model,
and how the new plan is merged with the calls already done. Pure: no model, no files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.engine.workflows.model import PlannedCall, Workflow
from core.engine.workflows.template import TemplateError, placeholders

NEW_ANSWER = "The user asked for a new answer. Answer again with the whole JSON value, matching the schema."
REFUSED = (
    "The tool {tool} refused the call {key}: {error}\n"
    "Answer again with the whole JSON value, matching the schema, and fix the problem."
)
DEPENDENT_NEW = (
    "An earlier answer that this one builds on was asked again, so it may have changed. "
    "Answer again with the whole JSON value, matching the schema, and keep it consistent with the new answer."
)
DEPENDENT_REFUSED = (
    "An earlier answer that this one builds on was corrected, because the tool {tool} refused the call {key}: "
    "{error}\n"
    "Answer again with the whole JSON value, matching the schema, and keep it consistent with the corrected answer."
)


def _names(*templates: Any) -> set[str]:
    found: set[str] = set()
    for template in templates:
        try:
            found |= placeholders(template)
        except TemplateError:  # a template that does not parse was refused when the workflow loaded
            continue
    return found


def model_deps(wf: Workflow, step_id: str) -> set[str]:
    """The model steps whose answers the step uses: through its arguments, its `over` list, or (for a model step)
    its prompt."""
    step = next((s for s in wf.steps if s.id == step_id), None)
    if step is None:
        return set()
    names = _names(step.prompt) if step.kind == "model" else _names(step.args)
    if step.over:
        names.add(step.over.split(".")[0])
    models = {s.id for s in wf.steps if s.kind == "model"}
    return (names & models) - {step_id}


def redo_targets(wf: Workflow, start: set[str]) -> list[str]:
    """`start` plus every later model step whose prompt uses a target, transitively, in workflow order."""
    targets = set(start)
    for step in wf.steps:
        if step.kind == "model" and step.id not in targets and model_deps(wf, step.id) & targets:
            targets.add(step.id)
    return [s.id for s in wf.steps if s.kind == "model" and s.id in targets]


def feedback(tool: str, key: str, error: str | None, dependent: bool = False) -> str:
    """The user message of a re-ask; `tool`, `key` and `error` are already redacted. A `dependent` step is asked again
    only because an earlier answer it builds on was asked again."""
    if dependent:
        if error is None:
            return DEPENDENT_NEW
        return DEPENDENT_REFUSED.format(tool=tool, key=key, error=error)
    if error is None:
        return NEW_ANSWER
    return REFUSED.format(tool=tool, key=key, error=error)


@dataclass(frozen=True)
class Merge:
    entries: list[dict[str, Any]]  # the new manifest "calls"
    pending: list[int]  # positions of the calls that run
    reused: int


def merge_calls(
    calls: Sequence[PlannedCall],
    saved: Sequence[dict[str, Any]],
    unchanged: Callable[[PlannedCall, dict[str, Any]], bool],
    fed: Callable[[PlannedCall, set[Path]], bool],
    written: Callable[[PlannedCall], set[Path]],
) -> Merge:
    """A saved call is reused when its key and digest are the plan's, it is done, its output is unchanged and nothing
    it is built from runs again; every other planned call is pending. Saved calls with no planned key are dropped."""
    by_key = {e["key"]: e for e in saved}
    entries: list[dict[str, Any]] = []
    pending: list[int] = []
    rewritten: set[Path] = set()
    for pos, call in enumerate(calls):
        old = by_key.get(call.key)
        if (
            old is not None
            and old["digest"] == call.digest
            and old["state"] == "done"
            and unchanged(call, old)
            and not fed(call, rewritten)
        ):
            entries.append(dict(old))
            continue
        entries.append(
            {
                "key": call.key,
                "tool": call.tool,
                "args": call.args,
                "digest": call.digest,
                "state": "pending",
                "summary": "",
                "output_sha256": None,
            }
        )
        pending.append(pos)
        rewritten |= written(call)
    return Merge(entries, pending, len(calls) - len(pending))
