"""A static scan of the project's files (makefile, manifests, CI, layout) for the `/init` prompt.

Nothing here starts a process or opens a file outside the workspace.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.agent.instructions import read_head

READ_LIMIT = 256 * 1024
MAX_SUBFOLDERS = 50
MAX_WORKFLOWS = 20
FACTS_SHARE = 0.2
SKIP_DIRS = frozenset({"node_modules", ".venv", "venv", "dist", "build", "target", "__pycache__"})
MAKEFILES = ("makefile", "Makefile", "GNUmakefile")
INTRO = (
    "Facts found by a scan of the project's files. "
    "Use these for every command; do not write a command that is not listed:"
)


@dataclass(frozen=True)
class MakeTarget:
    name: str
    comment: str  # "" when none


@dataclass(frozen=True)
class PythonProject:
    name: str | None
    requires_python: str | None
    scripts: tuple[str, ...]
    tools: tuple[str, ...]  # subset of ("pytest", "ruff", "ty", "uv"), in that order


@dataclass(frozen=True)
class NodeProject:
    name: str | None
    scripts: tuple[str, ...]
    manager: Literal["bun", "pnpm", "yarn", "npm"]


@dataclass(frozen=True)
class FolderFacts:
    path: str  # "" for the workspace, "cli" for a subfolder
    makefile: str | None  # the file name found, None when there is none
    targets: tuple[MakeTarget, ...]
    python: PythonProject | None
    node: NodeProject | None


@dataclass(frozen=True)
class CiCommand:
    run: str
    cwd: str | None


@dataclass(frozen=True)
class Workflow:
    file: str  # like ".github/workflows/ci.yml"
    commands: tuple[CiCommand, ...]


@dataclass(frozen=True)
class Facts:
    workspace: FolderFacts
    subfolders: tuple[FolderFacts, ...]  # name order; every scanned subfolder, also those with nothing in them
    workflows: tuple[Workflow, ...]
    more_subfolders: int  # real subfolders beyond MAX_SUBFOLDERS

    @property
    def empty(self) -> bool:
        """No makefile, python, node or workflow command anywhere."""
        folders = (self.workspace, *self.subfolders)
        return not any(f.makefile or f.python or f.node for f in folders) and not any(
            w.commands for w in self.workflows
        )

    def folder(self, path: str | None) -> FolderFacts | None:
        if path is None or path in ("", ".", "./"):
            return self.workspace
        name = path[2:] if path.startswith("./") else path
        name = name.rstrip("/")
        if name in ("", "."):
            return self.workspace
        for sub in self.subfolders:
            if sub.path == name:
                return sub
        return None


def _text(workspace_real: str, path: Path) -> str | None:
    """The decoded text of a regular file inside the workspace of at most READ_LIMIT bytes; else None."""
    # Opens the resolved path, so a link swapped in after the check is not followed on the last hop.
    # Not covered: a hard link inside the workspace to an outside file is read (undetectable by path;
    # accepted), and a link swapped into an intermediate folder between the check and the open (race remains).
    try:
        real = os.path.realpath(path)
        if os.path.commonpath([workspace_real, real]) != workspace_real:
            return None
        head = read_head(Path(real), READ_LIMIT + 1)
    except (OSError, ValueError):
        return None
    if head is None or len(head[0]) > READ_LIMIT:
        return None
    return head[0].decode("utf-8", errors="replace")


def _plain(name: str) -> bool:
    """A name that is safe to put in the prompt as is: one line, no whitespace or control character, at most 120."""
    return 0 < len(name) <= 120 and name == _clean(name) and not any(ch.isspace() for ch in name)


def _subfolders(workspace: Path) -> tuple[list[str], int]:
    names: list[str] = []
    try:
        with os.scandir(workspace) as entries:
            for entry in entries:
                if entry.name.startswith(".") or entry.name in SKIP_DIRS or not _plain(entry.name):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        names.append(entry.name)
                except OSError:
                    continue
    except OSError:
        return [], 0
    names.sort()
    return names[:MAX_SUBFOLDERS], max(len(names) - MAX_SUBFOLDERS, 0)


_ASSIGN = re.compile(r"^\s*(export\s+|override\s+)?[\w.\-]+\s*(\?=|\+=|::?=|=)")
_DIRECTIVES = frozenset(
    {
        *("include", "-include", "sinclude", "vpath", "export", "unexport", "override", "undefine"),
        *("ifeq", "ifneq", "ifdef", "ifndef", "else", "endif", "define", "endef"),
    }
)
_RULE = re.compile(r"^([^\s:#=][^:#=]*?)\s*::?(?!=)([^#]*)(#\s*(.*))?$")


def _makefile(text: str) -> tuple[MakeTarget, ...]:
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        if raw.endswith("\\"):
            pending += raw[:-1] + " "
            continue
        lines.append(pending + raw)
        pending = ""
    if pending:
        lines.append(pending)
    found: dict[str, MakeTarget] = {}
    define_depth = 0
    for line in lines:
        words = line.split()
        if words[:1] in (["override"], ["export"]) and words[1:2] == ["define"]:
            words = words[1:]
        if define_depth:
            define_depth += (words[:1] == ["define"]) - (words[:1] == ["endef"])
            continue
        if words[:1] == ["define"] and not line.startswith("\t"):
            define_depth = 1
            continue
        if not words or line.startswith("\t") or words[0].startswith("#") or words[0] in _DIRECTIVES or _ASSIGN.match(line):
            continue
        rule = _RULE.match(line)
        if rule is None:
            continue
        comment = _clean((rule.group(4) or "").lstrip("# ")) or ""
        for name in (_clean(n) for n in rule.group(1).split()):
            if not name or name.startswith(".") or "%" in name or "$" in name or name in found:
                continue
            found[name] = MakeTarget(name, comment)
    return tuple(found.values())


def _clean(value: object) -> str | None:
    """A fact string: one line, stripped, at most 120 characters; None for anything but a string."""
    if not isinstance(value, str):
        return None
    line = "".join(" " if ch.isspace() or not ch.isprintable() else ch for ch in value)
    return line.strip()[:120].strip()


def _names(table: object) -> tuple[str, ...]:
    if not isinstance(table, dict):
        return ()
    return tuple(name for key in table if (name := _clean(key)))


def _python(workspace_real: str, root: Path) -> PythonProject | None:
    text = _text(workspace_real, root / "pyproject.toml")
    if text is None:
        return None
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError, RecursionError):
        return None
    project = data.get("project")
    project = project if isinstance(project, dict) else {}
    tool = data.get("tool")
    tool = tool if isinstance(tool, dict) else {}

    def has(name: str, *files: str) -> bool:
        return name in tool or any(os.path.lexists(root / f) for f in files)

    tools = tuple(
        name
        for name, present in (
            ("pytest", has("pytest", "pytest.ini")),
            ("ruff", has("ruff", "ruff.toml", ".ruff.toml")),
            ("ty", has("ty", "ty.toml")),
            ("uv", has("uv", "uv.lock")),
        )
        if present
    )
    return PythonProject(
        _clean(project.get("name")),
        _clean(project.get("requires-python")),
        _names(project.get("scripts")),
        tools,
    )


_LOCKS = (
    (("bun.lock", "bun.lockb"), "bun"),
    (("pnpm-lock.yaml",), "pnpm"),
    (("yarn.lock",), "yarn"),
    (("package-lock.json",), "npm"),
)


def _node(workspace_real: str, root: Path) -> NodeProject | None:
    text = _text(workspace_real, root / "package.json")
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    manager: Literal["bun", "pnpm", "yarn", "npm"] = "npm"
    for files, name in _LOCKS:
        if any(os.path.lexists(root / f) for f in files):
            manager = name
            break
    return NodeProject(_clean(data.get("name")), _names(data.get("scripts")), manager)


def _folder(workspace_real: str, root: Path, path: str) -> FolderFacts:
    makefile = None
    targets: tuple[MakeTarget, ...] = ()
    for name in MAKEFILES:
        if os.path.lexists(root / name):
            text = _text(workspace_real, root / name)
            if text is not None:
                makefile, targets = name, _makefile(text)
                break
    return FolderFacts(path, makefile, targets, _python(workspace_real, root), _node(workspace_real, root))


_KEY = re.compile(r"^([\w.\-]+)\s*:\s*(.*)$")
_BLOCK = re.compile(r"^[|>][+-]?\d?\s*(#.*)?$")
MAX_CI_COMMANDS = 30


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _strip_comment(value: str) -> str:
    """`value` without a trailing ` # comment` that stands outside quotes."""
    quote = ""
    for i, ch in enumerate(value):
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1] in " \t"):
            return value[:i].rstrip()
    return value


def _workflow_commands(text: str) -> tuple[CiCommand, ...]:
    """The `run:` commands of a workflow, read by indentation (no YAML library)."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        body = raw.lstrip(" \t")
        if body and not body.startswith("#"):
            lines.append((len(raw) - len(body), body.rstrip()))

    commands: list[CiCommand] = []
    seen: set[tuple[str, str | None]] = set()
    in_jobs = False
    job_indent: int | None = None
    workflow_cwd: str | None = None
    job_cwd: str | None = None
    steps_indent: int | None = None  # the indent of the open `steps:` key
    step_indent: int | None = None
    step_cwd: str | None = None
    step_runs: list[str] = []

    def close() -> None:
        nonlocal step_indent, step_cwd, step_runs
        if step_indent is not None:
            cwd = step_cwd or job_cwd or workflow_cwd
            for run in step_runs:
                if len(commands) < MAX_CI_COMMANDS and (run, cwd) not in seen:
                    seen.add((run, cwd))
                    commands.append(CiCommand(run, cwd))
        step_indent, step_cwd, step_runs = None, None, []

    i = 0
    while i < len(lines):
        indent, body = lines[i]
        i += 1
        if indent == 0:
            close()
            in_jobs = body.startswith("jobs:")
            job_indent, job_cwd, steps_indent = None, None, None
            if not in_jobs:
                body_key = _KEY.match(body)
                if body_key and body_key.group(1) == "working-directory":
                    workflow_cwd = _clean(_unquote(body_key.group(2))) or None
            continue
        if in_jobs and job_indent is None:
            job_indent = indent
        if in_jobs and indent == job_indent and not body.startswith("- "):
            close()
            job_cwd, steps_indent = None, None
            continue
        if steps_indent is not None and (indent < steps_indent or (indent == steps_indent and not body.startswith("- "))):
            close()
            steps_indent = None
        if (
            body.startswith("- ")
            and steps_indent is not None
            and (step_indent is None or indent <= step_indent)
        ):
            close()
            step_indent = indent
            body = body[2:].lstrip()
            indent += 2
        elif step_indent is not None and indent <= step_indent:
            close()
        key = _KEY.match(body)
        if key is None:
            continue
        name, value = key.group(1), key.group(2).strip()
        if name == "working-directory":
            folder = _clean(_unquote(value)) or None
            if step_indent is not None:
                step_cwd = folder
            elif in_jobs:
                job_cwd = folder
            else:
                workflow_cwd = folder
        elif name == "steps" and not _strip_comment(value) and in_jobs:
            steps_indent = indent
        elif name == "run" and step_indent is not None:
            if _BLOCK.match(value):
                while i < len(lines) and lines[i][0] > indent:
                    if run := _clean(lines[i][1]):
                        step_runs.append(run)
                    i += 1
            elif value and (run := _clean(_unquote(_strip_comment(value)))):
                step_runs.append(run)
    close()
    return tuple(commands)


def _workflows(workspace_real: str, root: Path) -> tuple[Workflow, ...]:
    folder = root / ".github" / "workflows"
    try:
        if os.path.commonpath([workspace_real, os.path.realpath(folder)]) != workspace_real:
            return ()
        with os.scandir(folder) as entries:
            names = sorted(e.name for e in entries if e.name.endswith((".yml", ".yaml")) and _plain(e.name))
    except (OSError, ValueError):
        return ()
    found: list[Workflow] = []
    for name in names[:MAX_WORKFLOWS]:
        text = _text(workspace_real, folder / name)
        if text is not None:
            found.append(Workflow(f".github/workflows/{name}", _workflow_commands(text)))
    return tuple(found)


def scan(workspace: Path) -> Facts:
    """Read the project's files into a `Facts` value. Never raises, never runs a process."""
    try:
        real = os.path.realpath(workspace)
        names, more = _subfolders(Path(real))
        return Facts(
            _folder(real, Path(real), ""),
            tuple(_folder(real, Path(real) / n, n) for n in names),
            _workflows(real, Path(real)),
            more,
        )
    except Exception:
        return Facts(FolderFacts("", None, (), None, None), (), (), 0)


# Rendering. Every list has a cap; caps step down while the text does not fit the budget.

CAPS: tuple[int | None, ...] = (None, 20, 10, 5, 3, 1, 0)
# The groups in the order they are cut, with the lowest step each may reach.
_CUT_ORDER = (
    ("layout", 6),
    ("ci", 6),
    ("sub_scripts", 6),
    ("sub_targets", 6),
    ("project", 6),
    ("ws_scripts", 5),
    ("ws_targets", 5),
)


def _capped(entries: list[str], cap: int | None, extra_more: int = 0) -> list[str]:
    """The first `cap` entries and an `… and N more` line for the rest (None = all; 0 = nothing)."""
    if cap == 0 or (not entries and not extra_more):
        return []
    hidden = extra_more
    if cap is not None and len(entries) > cap:
        hidden += len(entries) - cap
        entries = entries[:cap]
    return [*entries, f"- … and {hidden} more"] if hidden else list(entries)


def _script_line(manager: str, folder: str, script: str) -> str:
    if not folder:
        return f"- {manager} run {script}"
    forms = {
        "bun": f"- bun run --cwd {folder} {script}",
        "yarn": f"- yarn --cwd {folder} run {script}",
        "pnpm": f"- pnpm --dir {folder} run {script}",
    }
    return forms.get(manager, f"- npm --prefix {folder} run {script}")


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _project_lines(folder: FolderFacts) -> list[str]:
    prefix = f"{folder.path}/" if folder.path else ""
    lines = []
    if (py := folder.python) is not None:
        info = [x for x in (py.name, f"Python {py.requires_python}" if py.requires_python else None) if x]
        if py.scripts:
            info.append(f"script {', '.join(f'`{n}`' for n in py.scripts)}")
        line = f"- {prefix}pyproject.toml" + (f": {', '.join(info)}" if info else "")
        configured = [t for t in py.tools if t != "uv"]
        if configured:
            line += f"; {_joined(configured)} configured"
        if "uv" in py.tools:
            line += "; uv"
        lines.append(line)
    if (node := folder.node) is not None:
        head = f"- {prefix}package.json: "
        lines.append(head + (f"{node.name}; " if node.name else "") + f"scripts run with {node.manager}")
    return lines


def _target_line(prefix: str, target: MakeTarget) -> str:
    return f"- make {prefix}{target.name}" + (f" — {target.comment}" if target.comment else "")


def _listed(targets: tuple[MakeTarget, ...]) -> list[MakeTarget]:
    """The targets worth showing a small model: a bare `help` says nothing."""
    return [t for t in targets if not (t.name == "help" and not t.comment)]


SUB_TARGETS_START = 10  # per subfolder, when the workspace has its own makefile


def _layout_line(folder: FolderFacts) -> str:
    held = [x for x in (folder.makefile, "pyproject.toml" if folder.python else None, "package.json" if folder.node else None) if x]
    return f"- {folder.path}/" + (f" ({', '.join(held)})" if held else "")


def render_facts(facts: Facts, budget_chars: int) -> str:
    """The facts block for the `/init` prompt, at most `budget_chars` long; "" when empty or nothing fits."""
    if facts.empty:
        return ""
    folders = (facts.workspace, *facts.subfolders)
    subs = facts.subfolders
    project = [line for f in folders for line in _project_lines(f)]
    ws_targets = [_target_line("", t) for t in _listed(facts.workspace.targets)]
    sub_lists = [[_target_line(f"-C {f.path} ", t) for t in _listed(f.targets)] for f in subs]
    per_folder = facts.workspace.makefile is not None

    def sub_target_lines(cap: int | None) -> list[str]:
        if not per_folder:
            return _capped([line for lines in sub_lists for line in lines], cap)
        if cap == 0:
            return []
        own = SUB_TARGETS_START if cap is None else min(cap, SUB_TARGETS_START)
        return [line for lines in sub_lists for line in _capped(lines, own)]
    ws_scripts = [_script_line(n.manager, "", s) for n in [facts.workspace.node] if n for s in n.scripts]
    sub_scripts = [_script_line(f.node.manager, f.path, s) for f in subs if f.node for s in f.node.scripts]
    ci = [(w.file, [f"- {c.run}" + (f" (in {c.cwd})" if c.cwd else "") for c in w.commands]) for w in facts.workflows]
    layout = [_layout_line(f) for f in subs]

    def build(steps: dict[str, int]) -> str:
        cap = {name: CAPS[steps[name]] for name in steps}
        sections: list[list[str]] = []
        if lines := _capped(project, cap["project"]):
            sections.append(["Project", *lines])
        if lines := [*_capped(ws_targets, cap["ws_targets"]), *sub_target_lines(cap["sub_targets"])]:
            sections.append(["Make targets", *lines])
        if lines := [*_capped(ws_scripts, cap["ws_scripts"]), *_capped(sub_scripts, cap["sub_scripts"])]:
            sections.append(["Package scripts", *lines])
        for file, commands in ci:
            if lines := _capped(commands, cap["ci"]):
                sections.append([f"CI ({file})", *lines])
        if lines := _capped(layout, cap["layout"], facts.more_subfolders):
            sections.append(["Layout", *lines])
        if not sections:
            return ""
        return "\n\n".join([INTRO, *("\n".join(section) for section in sections)])

    steps = {name: 0 for name, _ in _CUT_ORDER}
    while True:
        text = build(steps)
        if len(text) <= budget_chars:
            return text
        for name, floor in _CUT_ORDER:
            if steps[name] < floor:
                steps[name] += 1
                break
        else:
            return ""


# The check: make targets and package scripts named in an AGENTS.md against the facts.


@dataclass(frozen=True)
class CheckResult:
    checked: int  # distinct commands that could be checked
    missing: tuple[str, ...]  # distinct commands the facts do not contain, in file order


_SPAN = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_ENV_WORD = re.compile(r"^[A-Za-z_]\w*=")
_REDIRECT = re.compile(r"^(\d*|&)(>>?|<)")
_TARGET = re.compile(r"^[A-Za-z0-9_./-]+$")
_SCRIPT = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_MANAGERS = ("npm", "pnpm", "yarn", "bun")
_DIR_FLAGS = ("--cwd", "--prefix", "--dir", "-C")
_FILTER_FLAGS = (
    "--filter", "-F", "-w", "--workspace", "--workspaces", "-ws", "-r", "--recursive", "--include-workspace-root",
)  # fmt: skip
_NO_VALUE_FLAGS = (
    "--silent", "-s", "--if-present", "--watch", "--hot", "--bun", "--smol", "--parallel", "--stream",
    "--verbose", "--quiet", "-q",
)  # fmt: skip
_MAX_LINE = 2000
_HEREDOC = re.compile(r"(?<!<)<<(-?)\s*([\"']?)(\w+)\2")
_MAKEFILE_FLAGS = ("--file", "--makefile")
_MAKE_VALUE_FLAGS = (
    "--include-dir", "--old-file", "--assume-old", "--what-if",
    "--new-file", "--assume-new", "--load-average", "--max-load",
)  # fmt: skip
_MAKE_JOBS_FLAGS = ("-j", "--jobs")
_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".sh")
_PUNCTUATION = ".,;:)"
MAX_LISTED = 10


@dataclass(frozen=True)
class _Piece:
    text: str  # one code span, or the lines of one fenced block
    span: bool


def _code_pieces(text: str) -> list[_Piece]:
    """The contents of one-line code spans outside fences, and each fenced block as one piece."""
    pieces: list[_Piece] = []
    fence: str | None = None  # the opener's marks while inside a fence
    block: list[str] = []
    for line in text.splitlines():
        if fence is not None:
            stripped = line.rstrip(" ")
            marks = stripped.lstrip(" ")
            if len(stripped) - len(marks) <= 3 and len(marks) >= len(fence) and marks == fence[0] * len(marks):
                pieces.append(_Piece("\n".join(block), False))
                fence, block = None, []
            else:
                block.append(line)
        elif (m := _FENCE.match(line)) and not (m[1][0] == "`" and "`" in m[2]):
            fence = m[1]
        else:
            pieces.extend(_Piece(code, True) for code in _SPAN.findall(line))
    if fence is not None:
        pieces.append(_Piece("\n".join(block), False))
    return pieces


def _segments(line: str) -> list[tuple[str, str]]:
    """("cmd", text), ("(", "") and (")", "") in line order; not split inside quotes, cut at a comment."""
    out: list[tuple[str, str]] = []
    start = 0
    quote = ""
    i = 0
    n = len(line)

    def flush(end: int) -> None:
        if line[start:end].strip():
            out.append(("cmd", line[start:end]))

    while i < n:
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 1
            elif c == quote:
                quote = ""
        elif c == "\\":
            i += 1
        elif c in "'\"":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1].isspace()):
            break
        elif c in "()":
            flush(i)
            out.append((c, ""))
            start = i + 1
        else:
            width = 0
            if line[i : i + 2] in ("&&", "||"):
                width = 2
            elif c in ";|" or (c == "&" and not (line[i + 1 : i + 2] == ">" or (i and line[i - 1] == ">"))):
                width = 1
            if width:
                flush(i)
                start = i + width
                i += width
                continue
        i += 1
    flush(min(i, n))
    return out


def _words(command: str) -> list[str]:
    command = command.strip()
    if command.startswith("$ "):
        command = command[2:]
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    for i, word in enumerate(words):
        if _REDIRECT.match(word):
            words = words[:i]
            break
    while words and _ENV_WORD.match(words[0]):
        words.pop(0)
    return words


def _join(base: str | None, directory: str) -> str | None:
    """`directory` below the workspace-relative `base`; None when it cannot be placed."""
    if base is None or not directory or directory[0] in "/~" or "$" in directory:
        return None
    joined = posixpath.normpath(posixpath.join(base, directory))
    return "" if joined == "." else joined


def _has_alnum(word: str) -> bool:
    return any(c.isalnum() for c in word)


def _check_make(words: list[str], facts: Facts, cwd: str | None) -> list[tuple[str, bool]]:
    """(command, found) for each target of one `make` command; empty when it cannot be checked."""
    folder_path = cwd
    dirs: list[str] = []
    targets: list[str] = []
    i = 1
    while i < len(words):
        word = words[i]
        i += 1
        directory: str | None = None
        if word == "--directory":
            if i >= len(words):
                return []
            directory = words[i]
            i += 1
        elif word.startswith("--directory="):
            directory = word[len("--directory=") :]
        elif any(word == f or word.startswith(f + "=") for f in _MAKEFILE_FLAGS):
            return []
        elif word in _MAKE_VALUE_FLAGS:
            i += 1
        elif word in _MAKE_JOBS_FLAGS:
            if i < len(words) and words[i].isdigit():
                i += 1
        elif word.startswith("-") and not word.startswith("--"):
            for k, letter in enumerate(word[1:], 1):
                if letter not in "CfIoWlj":
                    if letter.isalpha():
                        continue
                    break
                attached = word[k + 1 :]
                if letter == "f":
                    return []
                if letter == "C":
                    if attached:
                        directory = attached
                    elif i < len(words):
                        directory = words[i]
                        i += 1
                    else:
                        return []
                elif not attached and (letter != "j" or (i < len(words) and words[i].isdigit())):
                    i += 1
                break
        elif word.startswith("-") or "=" in word or word.isdigit():
            continue
        elif _TARGET.match(word) and _has_alnum(word):
            targets.append(word)
        if directory is not None:
            dirs.append(directory)
            folder_path = _join(folder_path, directory)
    folder = facts.folder(folder_path) if folder_path is not None else None
    if folder is None:
        return []
    prefix = "make " + "".join(f"-C {d} " for d in dirs)
    known = {t.name for t in folder.targets}
    return [(prefix + target, target in known) for target in targets]


def _is_file(word: str) -> bool:
    return "/" in word or word.endswith(_SOURCE_SUFFIXES)


def _check_script(words: list[str], facts: Facts, cwd: str | None) -> list[tuple[str, bool]]:
    folder_path = cwd
    ran = False
    script: str | None = None
    end = 0
    i = 1
    while i < len(words):
        word = words[i]
        i += 1
        if word == "--":
            break
        flag, eq, value = word.partition("=")
        if flag in _FILTER_FLAGS and word.startswith("-") and not (words[0] == "bun" and flag == "-r"):
            return []
        if flag in _DIR_FLAGS and word.startswith("-"):
            if eq:
                folder_path = _join(folder_path, value)
            elif i < len(words):
                folder_path = _join(folder_path, words[i])
                i += 1
        elif word.startswith("-"):
            if (
                script is None
                and not eq
                and flag not in _NO_VALUE_FLAGS
                and i + 1 < len(words)
                and not words[i].startswith("-")
            ):
                return []
        elif not ran:
            if word != "run":
                return []
            ran = True
        elif script is None and _SCRIPT.match(word) and _has_alnum(word):
            if words[0] == "bun" and _is_file(word):
                return []
            script, end = word, i
    if script is None or folder_path is None:
        return []
    folder = facts.folder(folder_path)
    if folder is None:
        return []
    found = folder.node is not None and script in folder.node.scripts
    return [(" ".join(words[:end]), found)]


def _cd(words: list[str], cwd: str | None, facts: Facts) -> str | None:
    args = [w for w in words[1:] if w != "--" and not (w.startswith("-") and w != "-")]
    if not args or args[0] == "-":
        return None
    path = _join(cwd, args[0])
    if path is None or (path and facts.folder(path) is None):
        return None  # a folder the scan did not read stays unknown, however it is left
    return path


def _strip_punctuation(words: list[str]) -> None:
    if words and (stripped := words[-1].rstrip(_PUNCTUATION)):
        words[-1] = stripped


def _command_lines(piece: _Piece) -> list[str]:
    lines = [line for line in piece.text.split("\n") if len(line) <= _MAX_LINE]
    if not piece.span and any(line.lstrip(" ").startswith("$ ") for line in lines):
        lines = [line for line in lines if line.lstrip(" ").startswith("$ ")]
    return lines


def _check_piece(piece: _Piece, facts: Facts, seen: dict[str, bool]) -> None:
    cwd: str | None = ""
    saved: list[str | None] = []
    terminator: tuple[bool, str] | None = None  # (strip leading tabs, word) while in a here-document
    for line in _command_lines(piece):
        if terminator is not None:
            if (line.lstrip("\t") if terminator[0] else line).rstrip() == terminator[1]:
                terminator = None
            continue
        if heredoc := _HEREDOC.search(line):
            terminator = (bool(heredoc[1]), heredoc[3])
        for kind, command in _segments(line):
            if kind == "(":
                saved.append(cwd)
                continue
            if kind == ")":
                if saved:
                    cwd = saved.pop()
                continue
            words = _words(command)
            if not words:
                continue
            if words[0] == "cd":
                cwd = _cd(words, cwd, facts)
                continue
            if words[0] in ("pushd", "popd", "dirs"):
                cwd = None
                continue
            if words[0] not in ("make", *_MANAGERS) or any("#" in word for word in words):
                continue
            if piece.span:
                _strip_punctuation(words)
            results = _check_make(words, facts, cwd) if words[0] == "make" else _check_script(words, facts, cwd)
            for shown, found in results:
                seen.setdefault(shown, found)


def check_commands(text: str, facts: Facts) -> CheckResult:
    """The `make` and `<pm> run` commands in code spans and fences of `text` that `facts` does not contain."""
    try:
        seen: dict[str, bool] = {}
        for piece in _code_pieces(text):
            _check_piece(piece, facts, seen)
        return CheckResult(len(seen), tuple(c for c, found in seen.items() if not found))
    except Exception:
        return CheckResult(0, ())


def check_notice(result: CheckResult) -> tuple[Literal["info", "warn"], str] | None:
    if result.missing:
        shown = ", ".join(result.missing[:MAX_LISTED])
        extra = len(result.missing) - MAX_LISTED
        if extra > 0:
            shown += f" and {extra} more"
        return (
            "warn",
            f"/init check: not found by the scan: {shown}. They may still be right; check before relying on them.",
        )
    if result.checked >= 2:
        return "info", f"/init check: all {result.checked} make and script commands were found by the scan"
    if result.checked == 1:
        return "info", "/init check: the 1 make or script command was found by the scan"
    return None


def init_check(workspace: Path) -> tuple[Literal["info", "warn"], str] | None:
    """Read the workspace's AGENTS.md, scan again and report commands the scan did not find."""
    real = os.path.realpath(workspace)
    text = _text(real, Path(real) / "AGENTS.md")
    if text is None:
        return None
    return check_notice(check_commands(text, scan(workspace)))
