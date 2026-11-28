"""The `format` key of the settings files: which formatters may run after a write or edit. Reads two parsed objects;
the only I/O is the lookup of executables (the workspace's `node_modules/.bin`, then the absolute entries of PATH)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.engine.lsp.config import (
    Fingerprint,
    Source,
    _Builtin,
    _entry,
    _inside,
    _on,
    find_on_path,
    take_fingerprint,
)
from core.engine.text import escape

PLACEHOLDER = "{file}"


@dataclass(frozen=True)
class FormatterSpec:
    name: str
    argv: tuple[str, ...]  # the resolved command, then its arguments; `{file}` is replaced by the path
    extensions: frozenset[str]
    env: Mapping[str, str]
    source: Source
    fingerprint: Fingerprint | None = None  # an in-workspace program: what it was at launch, checked before each run
    evidence: str = ""  # what the repository supplies that this formatter runs (program or config sha256): part of what is trusted


@dataclass(frozen=True)
class FormatSetup:
    enabled: bool
    formatters: tuple[FormatterSpec, ...]
    missing: tuple[str, ...]  # built-ins whose program was not found (or whose config gate is closed)
    warnings: tuple[str, ...]  # launch warnings, escaped

    @staticmethod
    def off() -> FormatSetup:
        return FormatSetup(False, (), (), ())


_WEB = frozenset({".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".json", ".md", ".yaml", ".yml", ".html"})
BUILTINS: tuple[_Builtin, ...] = (
    _Builtin("prettier", "prettier", ("--write", PLACEHOLDER), _WEB),
    _Builtin("ruff", "ruff", ("format", PLACEHOLDER), frozenset({".py"})),
    _Builtin("gofmt", "gofmt", ("-w", PLACEHOLDER), frozenset({".go"})),
    _Builtin("rustfmt", "rustfmt", (PLACEHOLDER,), frozenset({".rs"})),
)


_CODE_RC = (".js", ".cjs", ".mjs", ".ts", ".cts", ".mts")
_DATA_RC = (".json", ".json5", ".yaml", ".yml", ".toml")
CONFIG_LIMIT = 1024 * 1024


def read_nofollow(path: Path, limit: int) -> bytes:
    """At most `limit` bytes of a regular file, never through a symlink (`core.engine.undo` cannot be imported here)."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        return os.read(fd, limit)
    finally:
        os.close(fd)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prettier_gate(workspace: Path) -> tuple[bool, tuple[str, ...]] | None:
    """The workspace's prettier config: None when there is none; else `(runs_code, evidence)`. A config runs code when it
    is a script (`prettier.config.*`, `.prettierrc.js` and kin), names `plugins`, or points at a module (a `prettier`
    string in package.json); anything unreadable counts as code. `evidence` is `<file> <sha256>` per such file, so a
    change to one changes what was trusted."""
    code = False
    evidence: list[str] = []
    found = False
    try:
        for entry in sorted(workspace.iterdir(), key=lambda e: e.name):
            name = entry.name
            if not (name == ".prettierrc" or name.startswith((".prettierrc.", "prettier.config."))) or not entry.is_file():
                continue
            found = True
            suffix = Path(name).suffix.lower()
            if name.startswith("prettier.config.") or suffix in _CODE_RC or (name != ".prettierrc" and suffix not in _DATA_RC):
                runs = True
            else:
                text = read_nofollow(entry, CONFIG_LIMIT + 1)
                runs = len(text) > CONFIG_LIMIT or b"plugins" in text.lower()
            if runs:
                code = True
                evidence.append(f"{name} {_sha(read_nofollow(entry, CONFIG_LIMIT + 1))}")
        package = workspace / "package.json"
        if package.is_file():
            raw = read_nofollow(package, CONFIG_LIMIT + 1)
            data = json.loads(raw.decode("utf-8")) if len(raw) <= CONFIG_LIMIT else None
            if isinstance(data, dict) and "prettier" in data:
                found = True
                value = data["prettier"]
                if not isinstance(value, dict) or "plugins" in value:
                    code = True
                    evidence.append(f"package.json {_sha(raw)}")
    except (OSError, ValueError):
        return (True, ("unreadable",)) if found else None
    return (code, tuple(evidence)) if found else None


def has_prettier_config(workspace: Path) -> bool:
    return prettier_gate(workspace) is not None


def _builtin(
    base: _Builtin, source: Source, workspace: Path, which: Callable[[str], str | None], warnings: list[str]
) -> FormatterSpec | None:
    gate_evidence: tuple[str, ...] = ()
    if base.name == "prettier":
        gate = prettier_gate(workspace)
        if gate is None:
            return None
        runs_code, gate_evidence = gate
        if runs_code:  # a config or plugin the repository supplies runs as code in prettier: the repository's, so trusted
            source = "project"
        local = workspace / "node_modules" / ".bin" / "prettier"
        try:
            usable = local.is_file()
        except OSError:
            usable = False
        if usable:  # a program the repository supplies is the repository's, whoever turned formatting on
            try:
                fp = take_fingerprint(str(local))
            except OSError:
                warnings.append(escape(f"format: {local} cannot be read; not used"))
                return None
            evidence = "\n".join([f"program {local} {fp.sha256}", *gate_evidence])
            return FormatterSpec(base.name, (str(local), *base.args), base.extensions, {}, "project", fp, evidence)
    found = which(base.executable)
    if not found:
        return None
    return FormatterSpec(base.name, (found, *base.args), base.extensions, {}, source, None, "\n".join(gate_evidence))


def read_format(
    global_data: Mapping[str, object] | None,
    project_data: Mapping[str, object] | None,
    workspace: Path,
    *,
    which: Callable[[str], str | None] = find_on_path,
) -> FormatSetup:
    g = (global_data or {}).get("format")
    p = (project_data or {}).get("format")
    merged = p if p is not None else g  # the project's value wins, as for other keys
    if not _on(merged):
        return FormatSetup.off()
    builtin_source: Source = "global" if _on(g) else "project"
    warnings: list[str] = []
    configured: dict[str, tuple[object, Source]] = {}
    for data, source in ((g, "global"), (p, "project")):
        servers = data.get("servers") if isinstance(data, dict) else None
        if servers is None:
            continue
        if not isinstance(servers, dict):
            warnings.append(escape(f"format ({source}): servers must be an object; ignored"))
            continue
        for name, raw in servers.items():
            if name in configured and configured[name][0] is False:
                continue  # a `false` from any source stays off: a later source cannot switch it back on
            configured[name] = (False if raw is False else raw, source)
    table: list[tuple[str, Source, _Builtin | None, object]] = []
    seen: set[str] = set()
    for b in BUILTINS:
        seen.add(b.name)
        raw, source = configured.get(b.name, (None, builtin_source))
        table.append((b.name, source, b, raw))
    table.extend((n, s, None, r) for n, (r, s) in configured.items() if n not in seen)
    specs: list[FormatterSpec] = []
    missing: list[str] = []
    for name, source, base, raw in table:
        if raw is False:
            continue
        if raw is None:
            assert base is not None
            spec = _builtin(base, source, workspace, which, warnings)
            if spec is None:
                missing.append(name)
                continue
            specs.append(spec)
            continue
        entry, problem = _entry(name, raw, base, source, workspace, which)
        if entry is None:
            warnings.append(escape(f'format ({source}): servers."{name}": {problem}; skipped'))
            continue
        fingerprint = None
        if _inside(Path(entry.command), workspace):
            try:
                fingerprint = take_fingerprint(entry.command)
            except OSError:
                warnings.append(escape(f'format ({source}): servers."{name}": cannot read {entry.command}; skipped'))
                continue
        evidence = f"program {entry.command} {fingerprint.sha256}" if fingerprint is not None else ""
        specs.append(FormatterSpec(name, (entry.command, *entry.args), entry.extensions, entry.env, entry.source, fingerprint, evidence))
    return FormatSetup(True, tuple(specs), tuple(missing), tuple(warnings))


def project_needs_trust(setup: FormatSetup) -> bool:
    """Does the project file start a formatter that will actually run? Such a file waits for project trust."""
    return any(spec.source == "project" for spec in setup.formatters)


def trust_evidence(setup: FormatSetup) -> bytes:
    """What the repository supplies to project-sourced formatters (programs, code configs), as bytes that are appended
    to the project settings' bytes in what the user trusts: a changed program or config asks again. Empty when none."""
    lines = sorted(f"{s.name}\0{s.evidence}" for s in setup.formatters if s.source == "project" and s.evidence)
    return ("\n\0formatters\0\n" + "\n".join(lines)).encode() if lines else b""
