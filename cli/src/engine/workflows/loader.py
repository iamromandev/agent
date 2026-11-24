"""Finding and reading workflows (built-in, global, project). Files come from a repository: nothing in them is run."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.engine.commands.loader import NAME
from src.engine.fileio import parse_json, read_regular
from src.engine.permissions.setup import MAX_WARNINGS
from src.engine.text import escape
from src.engine.workflows.model import Source, Step, Workflow
from src.engine.workflows.schema import check_schema
from src.engine.workflows.template import TemplateError, placeholders

MAX_BYTES = 64 * 1024
MAX_STEPS = 20
FORBIDDEN_TOOLS = frozenset({"bash", "task", "todo", "skill", "term_start", "term_stop", "term_read"})
DEFAULT_FOLDER = "workflows/{name}/{run}"
STEP_ID = re.compile(r"[a-z_][a-z0-9_]*")
RESERVED_IDS = frozenset({"input", "folder", "run", "name", "index", "item"})
BASE_NAMES = frozenset({"input", "folder", "run", "name"})
FOLDER_NAMES = frozenset({"input", "run", "name"})
TOP_KEYS = frozenset({"description", "input", "folder", "steps"})
STEP_KEYS: dict[str, frozenset[str]] = {
    "model": frozenset({"id", "kind", "prompt", "schema"}),
    "tool": frozenset({"id", "kind", "tool", "args"}),
    "each": frozenset({"id", "kind", "over", "tool", "args"}),
}
OVER = re.compile(r"[a-z_][a-z0-9_]*(?:\.[A-Za-z0-9_]+)+")


@dataclass
class Workflows:
    items: dict[str, Workflow] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # already escaped and capped


def builtin_dir() -> Path:
    """The workflows shipped with the agent: `<this package>/builtin`."""
    return Path(__file__).resolve().parent / "builtin"


def global_dir(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/workflows`, else `~/.config/agent/workflows`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "workflows"


def project_dir(workspace: Path) -> Path:
    return workspace / ".agent" / "workflows"


def _check_folder(folder: object) -> str | None:
    if not isinstance(folder, str) or not folder:
        return "folder must be a non-empty string"
    if folder.startswith(("/", "~")):
        return "folder must be relative to the workspace"
    parts = folder.split("/")
    if ".." in parts:
        return "folder may not contain a '..' segment"
    if any(p.casefold() in (".agent", ".git") for p in parts):
        return "folder may not be under .agent or .git"
    try:
        unknown = placeholders(folder) - FOLDER_NAMES
    except TemplateError as exc:
        return f"folder: {exc}"
    if unknown:
        return f"folder: unknown name '{sorted(unknown)[0]}'"
    if "run" not in placeholders(folder):
        return "folder must contain {run}, so every run has its own folder"
    return None


def _check_names(label: str, value: Any, allowed: frozenset[str], *, local: bool) -> str | None:
    """Every name a template uses must be in `allowed`; `item` and `index` only where `local` says they exist."""
    try:
        used = placeholders(value)
    except TemplateError as exc:
        return f"{label}: {exc}"
    for name in sorted(used):
        if name in ("item", "index") and not local:
            return f"{label}: {{{name}}} is only available in an each step or $map"
        if name not in allowed and name not in ("item", "index"):
            return f"{label}: unknown name '{name}'"
    return None


def _check_tool(step: Step) -> str | None:
    if step.tool in FORBIDDEN_TOOLS or step.tool.startswith("mcp__"):
        return f"step '{step.id}': tool '{step.tool}' is not allowed in a workflow"
    return None


def _step(raw: object, seen: dict[str, Step], phase: list[str]) -> Step | str:
    """One step, or why it is not valid. `seen` holds the earlier steps by id; `phase` has "tools" once one began."""
    if not isinstance(raw, dict):
        return "a step must be an object"
    step_id = raw.get("id")
    if not isinstance(step_id, str) or not STEP_ID.fullmatch(step_id) or step_id in RESERVED_IDS:
        return f"bad step id {step_id!r} (use [a-z_][a-z0-9_]*, not a reserved name)"
    if step_id in seen:
        return f"duplicate step id '{step_id}'"
    kind = raw.get("kind")
    if kind not in STEP_KEYS:
        return f"step '{step_id}': kind must be model, tool or each"
    for key in raw:
        if key not in STEP_KEYS[kind]:
            return f"step '{step_id}': unknown key '{key}'"
    models = frozenset(s for s, st in seen.items() if st.kind == "model")
    allowed = BASE_NAMES | models
    if kind == "model":
        if phase:
            return f"model step '{step_id}' comes after a tool step"
        prompt, schema = raw.get("prompt"), raw.get("schema")
        if not isinstance(prompt, str) or not prompt.strip():
            return f"step '{step_id}': prompt must be a non-empty string"
        if (error := _check_names(f"step '{step_id}' prompt", prompt, allowed, local=False)) is not None:
            return error
        if (error := check_schema(schema)) is not None:
            return f"step '{step_id}' schema: {error}"
        assert isinstance(schema, dict)
        return Step(step_id, "model", prompt=prompt, schema=schema)
    phase.append("tools")
    tool, args = raw.get("tool"), raw.get("args", {})
    if not isinstance(tool, str) or not tool:
        return f"step '{step_id}': tool must be a non-empty string"
    if not isinstance(args, dict):
        return f"step '{step_id}': args must be an object"
    over = ""
    if kind == "each":
        over = raw.get("over")  # type: ignore[assignment]
        if not isinstance(over, str) or not OVER.fullmatch(over) or over.split(".")[0] not in models:
            return f"step '{step_id}': over must be '<model step>.<path>' naming an earlier model step"
    step = Step(step_id, kind, tool=tool, args=args, over=over)
    if (error := _check_tool(step)) is not None:
        return error
    return _check_names(f"step '{step_id}' args", args, allowed, local=kind == "each") or step


def parse_workflow(name: str, source: Source, path: Path, data: bytes) -> Workflow:
    """The workflow in `data`. Raises `ValueError` with the reason when it is not valid."""
    try:
        raw = parse_json(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"not UTF-8: {exc.reason}") from None
    if not isinstance(raw, dict):
        raise ValueError("a workflow must be a JSON object")
    for key in raw:
        if key not in TOP_KEYS:
            raise ValueError(f"unknown key '{key}'")
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    help_text = ""
    if "input" in raw:
        block = raw["input"]
        if not isinstance(block, dict) or set(block) - {"description"} or not isinstance(block.get("description", ""), str):
            raise ValueError("input must be {\"description\": <text>}")
        help_text = block.get("description", "")
    folder = raw.get("folder", DEFAULT_FOLDER)
    if (error := _check_folder(folder)) is not None:
        raise ValueError(error)
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_STEPS:
        raise ValueError(f"steps must be a list of 1 to {MAX_STEPS} steps")
    seen: dict[str, Step] = {}
    phase: list[str] = []
    for item in raw_steps:
        step = _step(item, seen, phase)
        if isinstance(step, str):
            raise ValueError(step)
        seen[step.id] = step
    return Workflow(
        name, source, path, hashlib.sha256(data).hexdigest(), description, help_text, str(folder), tuple(seen.values())
    )


def _read_file(path: Path, source: Source) -> bytes:
    if source != "project":
        return read_regular(path, MAX_BYTES)
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return read_regular(Path(path.name), MAX_BYTES, follow=False, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _load_dir(directory: Path, source: Source, found: dict[str, Workflow], warnings: list[str]) -> None:
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        warnings.append(f"{directory}: {exc.strerror or exc}")
        return
    for path in entries:
        if path.suffix != ".json":
            continue
        if source == "project" and path.is_symlink():
            warnings.append(f"{path}: a symlink is not followed in the project folder")
        elif not NAME.fullmatch(path.stem):
            warnings.append(f"{path}: not a valid workflow name")
        else:
            try:
                found[path.stem] = parse_workflow(path.stem, source, path, _read_file(path, source))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
                warnings.append(f"{path}: {reason}")


def load_workflows(workspace: Path, env: Mapping[str, str] = os.environ, builtin: Path | None = None) -> Workflows:
    """Read the folders once. Later sources replace earlier ones by name: builtin, global, project. Problems become
    warnings. `builtin` is the shipped folder (`builtin_dir()`) unless a test names another."""
    found: dict[str, Workflow] = {}
    warnings: list[str] = []
    _load_dir(builtin if builtin is not None else builtin_dir(), "builtin", found, warnings)
    _load_dir(global_dir(env), "global", found, warnings)
    project = project_dir(workspace)
    if project.parent.is_symlink() or project.is_symlink():
        warnings.append(f"{project}: a symlink is not followed in the project folder")
    else:
        _load_dir(project, "project", found, warnings)
    shown = [escape(w) for w in warnings]
    if len(shown) > MAX_WARNINGS:
        shown = [*shown[:MAX_WARNINGS], f"and {len(shown) - MAX_WARNINGS} more warnings"]
    return Workflows(found, shown)
