"""The `lsp` key of the settings files: which language servers may run. Reads two parsed objects; the only I/O is
the lookup of executables (`.venv/bin` under a rule, then the absolute entries of PATH)."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.engine.checkpoints.git import find_git
from core.engine.mentions_index import git_env
from core.engine.text import escape

Source = Literal["builtin", "global", "project"]
GIT_LIMIT = 5.0
MAX_PROGRAM_BYTES = 64 * 1024 * 1024  # a `.venv/bin` program is fingerprinted by reading it whole: at most this big


@dataclass(frozen=True)
class Fingerprint:
    """A program as it was at launch: the real file behind the path (symlinks resolved), its sha256 and
    `(st_dev, st_ino, st_size, st_mtime_ns)`."""

    path: str
    sha256: str
    stat: tuple[int, int, int, int]


def take_fingerprint(command: str) -> Fingerprint:
    """Read the file `command` names (through symlinks) once, bounded by `MAX_PROGRAM_BYTES`. Raises OSError (also
    for a file that is too big or not a regular file)."""
    real = os.path.realpath(command)
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        if info.st_size > MAX_PROGRAM_BYTES:
            raise OSError("too big")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(fd, 1 << 20):
            total += len(chunk)
            if total > MAX_PROGRAM_BYTES:
                raise OSError("too big")
            digest.update(chunk)
    finally:
        os.close(fd)
    return Fingerprint(real, digest.hexdigest(), (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))


def changed_since_launch(command: str, expected: Fingerprint) -> bool:
    """Is the program at `command` not the one that was fingerprinted? (blocking: reads the file)"""
    try:
        return take_fingerprint(command) != expected
    except OSError:
        return True


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    extensions: frozenset[str]
    env: Mapping[str, str]
    source: Source
    fingerprint: Fingerprint | None = None  # a `.venv/bin` program: what it was at launch, checked again before it starts


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
    """`<workspace>/.venv/bin/<name>` programs: usable only when the workspace is itself the top of a git repository
    (not a folder inside a parent repo) in which nothing under `.venv` is tracked (a cloned repo could commit its
    own `.venv/bin/ty`). An untracked one was made by the user's own tooling, so it takes the source of the file that
    turned LSP on; its bytes are fingerprinted at launch and checked again before it starts. Git is asked once
    (two calls), and only when some program exists there. Every git call ignores the repository's own `core.fsmonitor`."""

    def __init__(self, workspace: Path, warnings: list[str]) -> None:
        self._workspace = workspace
        self._warnings = warnings
        self._clean: bool | None = None

    def _git(self, git: str, *args: str) -> subprocess.CompletedProcess[bytes] | None:
        try:
            return subprocess.run(
                [git, "-c", "core.fsmonitor=false", *args],
                cwd=self._workspace,
                env=git_env(),
                capture_output=True,
                timeout=GIT_LIMIT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _untracked(self) -> bool:
        git = find_git(os.environ, self._workspace)
        if git is None:
            return False
        top = self._git(git, "rev-parse", "--show-toplevel")
        if top is None or top.returncode != 0:
            return False
        try:
            if Path(top.stdout.decode().strip()).resolve() != self._workspace.resolve():
                return False  # a workspace inside a parent repository: that repo's `.venv` rules are not ours to read
        except (OSError, RuntimeError, UnicodeDecodeError):
            return False
        done = self._git(git, "ls-files", "--", ".venv")
        return done is not None and done.returncode == 0 and not done.stdout.strip()

    def executable(self, name: str) -> tuple[str, Fingerprint] | None:
        candidate = self._workspace / ".venv" / "bin" / name
        try:
            if not (candidate.is_file() and os.access(candidate, os.X_OK)):
                return None
        except OSError:
            return None
        if self._clean is None:
            self._clean = self._untracked()
        if not self._clean:
            return None
        try:
            return str(candidate), take_fingerprint(str(candidate))
        except OSError as exc:
            limit = MAX_PROGRAM_BYTES // (1024 * 1024)
            why = f"is over {limit} MiB" if str(exc) == "too big" else "cannot be read"
            self._warnings.append(escape(f"lsp: {candidate} {why}; not used"))
            return None


def _inside(path: Path, workspace: Path) -> bool:
    """Does `path` lie in the workspace once symlinks are resolved on both sides?"""
    try:
        return path.resolve().is_relative_to(workspace.resolve())
    except (OSError, RuntimeError):
        return True  # cannot tell: treat it as the repository's


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
        if _inside(target, workspace):  # a program the repository supplies is the repository's, whoever named it
            source = "project"
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
    venvs = _Venvs(workspace, warnings)
    checker_taken = False
    for name, source, base, raw in table:
        if raw is False:
            continue
        if raw is None:  # a built-in as it is: used only when its program is found
            assert base is not None
            on_path = which(base.executable)
            venv = None if on_path else venvs.executable(base.executable)
            if on_path:
                spec: ServerSpec | None = ServerSpec(name, on_path, base.args, base.extensions, {}, source)
            elif venv:
                spec = ServerSpec(name, venv[0], base.args, base.extensions, {}, source, venv[1])
            else:
                continue
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
    return LspSetup(True, warnings_flag, tuple(specs), tuple(warnings))


def project_needs_trust(
    global_data: Mapping[str, object] | None,
    project_data: Mapping[str, object] | None,
    workspace: Path,
    *,
    which: Callable[[str], str | None] = find_on_path,
) -> bool:
    """Does the project file start a language server that will actually run: a project-sourced spec in what
    `read_lsp` builds from both files? Such a file waits for the same trust as project allow rules and hooks."""
    return any(spec.source == "project" for spec in read_lsp(global_data, project_data, workspace, which=which).servers)
