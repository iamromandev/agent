"""The `lsp` key of the settings files: which language servers may run. Reads two parsed objects; the only I/O is
the lookup of executables (`.venv/bin` under a rule, then the absolute entries of PATH)."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.engine.checkpoints.git import find_git
from core.engine.text import escape

Source = Literal["builtin", "global", "project"]
GIT_LIMIT = 5.0
PROJECT_UNSUPPORTED = "project lsp servers are not supported yet; ignored: "


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    extensions: frozenset[str]
    env: Mapping[str, str]
    source: Source


@dataclass(frozen=True)
class LspSetup:
    enabled: bool
    warnings: bool  # also report warning-level diagnostics
    servers: tuple[ServerSpec, ...]
    warnings_text: tuple[str, ...]  # launch warnings, escaped

    @staticmethod
    def off() -> LspSetup:
        return LspSetup(False, False, (), ())


@dataclass(frozen=True)
class _Builtin:
    name: str
    executable: str
    args: tuple[str, ...]
    extensions: frozenset[str]


_PY = frozenset({".py", ".pyi"})
_TS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
BUILTINS: tuple[_Builtin, ...] = (
    _Builtin("ty", "ty", ("server",), _PY),
    _Builtin("basedpyright", "basedpyright-langserver", ("--stdio",), _PY),
    _Builtin("pyright", "pyright-langserver", ("--stdio",), _PY),
    _Builtin("ruff", "ruff", ("server",), _PY),
    _Builtin("typescript", "typescript-language-server", ("--stdio",), _TS),
    _Builtin("gopls", "gopls", (), frozenset({".go"})),
    _Builtin("rust-analyzer", "rust-analyzer", (), frozenset({".rs"})),
)
CHECKERS = ("ty", "basedpyright", "pyright")


def find_on_path(name: str) -> str | None:
    """`name` on the absolute entries of PATH only (an empty or relative entry means the current folder)."""
    entries = [e for e in os.environ.get("PATH", "").split(os.pathsep) if e and os.path.isabs(e)]
    return shutil.which(name, path=os.pathsep.join(entries)) if entries else None


class _Venvs:
    """`<workspace>/.venv/bin/<name>` programs: usable only when the workspace is a git repository in which nothing
    under `.venv` is tracked (a cloned repo could commit its own `.venv/bin/ty`). Such a program counts as
    project-sourced. Git is asked at most once, and only when some program exists there."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._clean: bool | None = None

    def _untracked(self) -> bool:
        git = find_git(os.environ, self._workspace)
        if git is None:
            return False
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
        try:
            done = subprocess.run(
                [git, "ls-files", "--", ".venv"], cwd=self._workspace, env=env, capture_output=True, timeout=GIT_LIMIT, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0 and not done.stdout.strip()

    def executable(self, name: str) -> str | None:
        candidate = self._workspace / ".venv" / "bin" / name
        try:
            if not (candidate.is_file() and os.access(candidate, os.X_OK)):
                return None
        except OSError:
            return None
        if self._clean is None:
            self._clean = self._untracked()
        return str(candidate) if self._clean else None


def _on(value: object) -> bool:
    if value is True:
        return True
    return isinstance(value, dict) and value.get("enabled") is not False


def _entry(
    name: str, raw: object, base: _Builtin | None, source: Source, workspace: Path, which: Callable[[str], str | None]
) -> tuple[ServerSpec | None, str | None]:
    if not isinstance(raw, dict):
        return None, "must be an object or false"
    command = raw.get("command", base.executable if base else None)
    if not isinstance(command, str) or not command.strip():
        return None, "needs a non-empty command"
    args_raw = raw.get("args", list(base.args) if base else [])
    if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
        return None, "args must be a list of strings"
    exts_raw = raw.get("extensions", sorted(base.extensions) if base else None)
    if not isinstance(exts_raw, list) or not exts_raw or not all(isinstance(e, str) and e.startswith(".") and len(e) > 1 for e in exts_raw):
        return None, 'extensions must be a non-empty list of strings that start with "."'
    env_raw = raw.get("env", {})
    if not isinstance(env_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()):
        return None, "env must be an object of strings"
    if os.sep in command:  # a path: it must be there (a relative one is relative to the workspace, the spawn's cwd)
        target = Path(command) if os.path.isabs(command) else workspace / command
        try:
            usable = target.is_file() and os.access(target, os.X_OK)
        except OSError:
            usable = False
        if not usable:
            return None, f"command {command} is not an executable file"
        resolved = str(target)
    else:  # a bare name: only an absolute PATH entry may supply it, never the spawn's own lookup
        found = which(command)
        if found is None:
            return None, f"command {command} was not found on PATH"
        resolved = found
    return ServerSpec(name, resolved, tuple(args_raw), frozenset(e.lower() for e in exts_raw), dict(env_raw), source), None


def read_lsp(
    global_data: Mapping[str, object] | None,
    project_data: Mapping[str, object] | None,
    workspace: Path,
    *,
    which: Callable[[str], str | None] = find_on_path,
) -> LspSetup:
    g = (global_data or {}).get("lsp")
    p = (project_data or {}).get("lsp")
    merged = p if p is not None else g  # the project's value wins, as for other keys
    if not _on(merged):
        return LspSetup.off()
    # a built-in has the source of the file that turned LSP on
    builtin_source: Source = "global" if _on(g) else "project"
    options: dict[str, object] = {}
    for part in (g, p):
        if isinstance(part, dict):
            options.update(part)
    warnings_flag = options.get("warnings") is True
    warnings: list[str] = []
    configured: dict[str, tuple[object, Source]] = {}
    ignored: list[str] = []
    for data, source in ((g, "global"), (p, "project")):
        servers = data.get("servers") if isinstance(data, dict) else None
        if servers is None:
            continue
        if not isinstance(servers, dict):
            warnings.append(escape(f"lsp ({source}): servers must be an object; ignored"))
            continue
        for name, raw in servers.items():
            if raw is False:
                configured[name] = (False, source)  # a switch-off only ever tightens
            elif source == "project":
                ignored.append(name)  # phase 1: no project servers (the entry is not even read)
            else:
                configured[name] = (raw, source)
    table: list[tuple[str, Source, _Builtin | None, object]] = []
    seen: set[str] = set()
    for b in BUILTINS:
        seen.add(b.name)
        raw, source = configured.get(b.name, (None, builtin_source))
        table.append((b.name, source, b, raw))
    for name, (raw, source) in configured.items():
        if name not in seen:
            table.append((name, source, None, raw))
    specs: list[ServerSpec] = []
    venvs = _Venvs(workspace)
    checker_taken = False
    for name, source, base, raw in table:
        if raw is False:
            continue
        if raw is None:  # a built-in as it is: used only when its program is found
            assert base is not None
            path = which(base.executable)
            if builtin_source == "project":  # only the project file turned LSP on
                if path is not None or venvs.executable(base.executable) is not None:
                    ignored.append(name)
                continue
            if path is None:
                if venvs.executable(base.executable) is not None:
                    ignored.append(name)  # only a project-sourced `.venv` program exists: not supported yet
                continue
            spec: ServerSpec | None = ServerSpec(name, path, base.args, base.extensions, {}, source)
        else:
            spec, problem = _entry(name, raw, base, source, workspace, which)
            if spec is None:
                warnings.append(escape(f'lsp ({source}): servers."{name}": {problem}; skipped'))
                continue
        if name in CHECKERS:
            if checker_taken:
                continue
            checker_taken = True
        specs.append(spec)
    if ignored:
        warnings.append(escape(PROJECT_UNSUPPORTED + ", ".join(dict.fromkeys(ignored))))
    return LspSetup(True, warnings_flag, tuple(specs), tuple(warnings))
