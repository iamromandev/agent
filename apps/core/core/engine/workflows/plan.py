"""Turning a workflow and the model's answers into the exact list of tool calls, checked before anything runs."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.engine.guardrails.setup import guard_of
from core.engine.permissions.setup import Rules
from core.engine.text import escape
from core.engine.tools.base import Tool, ToolContext, check_args
from core.engine.tools.paths import InvalidPath, resolve_path
from core.engine.workflows.manifest import canonical
from core.engine.workflows.model import PlannedCall, Step, Workflow
from core.engine.workflows.template import Scope, TemplateError, render


class PlanError(Exception):
    """The plan cannot be run; the message says why."""


def digest(args: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(args).encode("utf-8")).hexdigest()


def _paths(value: Any) -> list[Any]:
    """Every value stored under a key named `path`, at any depth."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path":
                found.append(item)
            found.extend(_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_paths(item))
    return found


INPUT_KEYS = frozenset({"image", "audio", "music", "mask", "references"})
CLIPBOARD = "clipboard:"
GUARDED = (".agent", ".git")


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _flatten(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def input_refs(value: Any) -> list[Any]:
    """Every leaf stored under a key named like a file input (`image`, `audio`, `music`, `mask`, `references`), at
    any depth and inside lists and objects, so `assemble_video` scenes are covered. Not-strings are returned as they
    are: the check refuses them."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in INPUT_KEYS:
                found.extend(_flatten(item))
            else:
                found.extend(input_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(input_refs(item))
    return found


def _guarded(path: Path, root: Path) -> bool:
    return any(p.casefold() in GUARDED for p in path.relative_to(root).parts)


def _input_ok(raw: Any, ctx: ToolContext, folder: Path, lexical_folder: Path) -> bool:
    """The input rule: a file input is a string that is `clipboard:N` (the tool checks it) or resolves (symlinks
    followed) inside the workspace, not under `.agent` or `.git`, not a protected path; and a path that is written
    inside the run folder (`{folder}/...`, before symlinks) must still resolve inside it, so a link in the run
    folder cannot lead a workflow output out. Other workspace files (an asset the workflow names) are allowed."""
    if not isinstance(raw, str):
        return False
    if raw.startswith(CLIPBOARD):
        return True
    try:
        path = resolve_path(ctx, raw)
    except InvalidPath:
        return False
    root = ctx.workspace.resolve()
    if not path.is_relative_to(root) or _guarded(path, root) or guard_of(ctx).paths.is_protected(path):
        return False
    written = Path(os.path.normpath(ctx.workspace / raw)) if not raw.startswith(("/", "~")) else None
    if written is not None and written.is_relative_to(lexical_folder):
        return path.is_relative_to(folder)
    return True


def _inside(raw: Any, ctx: ToolContext, folder: Path) -> bool:
    if not isinstance(raw, str):
        return False
    try:
        path = resolve_path(ctx, raw)
    except InvalidPath:
        return False
    return path != folder and path.is_relative_to(folder)


def _check(
    key: str, step: Step, args: Any, tools: Mapping[str, Tool], ctx: ToolContext, rules: Rules, folder: Path, lexical: Path
) -> dict[str, Any]:
    tool = tools.get(step.tool)
    if tool is None:
        raise PlanError(f"tool {step.tool} is not available")
    if not isinstance(args, dict):
        raise PlanError(f"{key}: args are not an object")
    if (error := check_args(tool.parameters, args)) is not None:
        raise PlanError(f"{key}: {step.tool}: {error}")
    if (verdict := rules.deny(step.tool, args, tool.grant_arg)) is not None:
        raise PlanError(f"{key}: denied by rule {escape(verdict.rule.text)} ({verdict.source})")
    if not all(_inside(raw, ctx, folder) for raw in _paths(args)):
        raise PlanError(f"{key}: path leaves the run folder")
    if not all(_input_ok(raw, ctx, folder, lexical) for raw in input_refs(args)):
        raise PlanError(f"{key}: an input file is outside the workspace, under .agent or .git, or protected")
    return args


def plan_calls(
    wf: Workflow, scope: Scope, tools: Mapping[str, Tool], ctx: ToolContext, rules: Rules, folder: Path
) -> list[PlannedCall]:
    """Every call of the tool and `each` steps, in order, rendered from `scope` and checked: the tool exists, its
    arguments are valid, no deny rule refuses it, every `path` is inside `folder` (resolved, so a symlink out is
    caught) and every file input obeys `_input_ok`. Raises `PlanError`; nothing has run yet."""
    lexical = Path(os.path.normpath(folder))
    folder = folder.resolve()
    calls: list[PlannedCall] = []
    for step in wf.steps:
        if step.kind == "model":
            continue
        try:
            if step.kind == "tool":
                rendered = [(step.id, None, render(step.args, scope))]
            else:
                items = render("{" + step.over + "}", scope)
                if not isinstance(items, list):
                    raise PlanError(f"step {step.id}: {step.over} is not a list")
                rendered = [
                    (f"{step.id}/{n}", n, render(step.args, Scope({**scope.values, "item": item, "index": n})))
                    for n, item in enumerate(items, 1)
                ]
        except TemplateError as exc:
            raise PlanError(f"step {step.id}: {exc}") from exc
        for key, index, args in rendered:
            checked = _check(key, step, args, tools, ctx, rules, folder, lexical)
            calls.append(PlannedCall(key, step.id, index, step.tool, checked, digest(checked)))
    return calls
