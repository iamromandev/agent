"""The settings files: where they live and how they are read. A bad file is a value (`RuleFile.error`), not an exception."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.fileio import parse_json, read_regular
from src.engine.permissions.rules import Rule, RuleError, parse_rule


def global_settings_path(env: Mapping[str, str] = os.environ) -> Path:
    """`$XDG_CONFIG_HOME/agent/settings.json`, else `~/.config/agent/settings.json`. A relative XDG_CONFIG_HOME is ignored."""
    config_home = env.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home and Path(config_home).is_absolute() else Path.home() / ".config"
    return base / "agent" / "settings.json"


def project_settings_path(workspace: Path) -> Path:
    """`<workspace>/.agent/settings.json`."""
    return workspace / ".agent" / "settings.json"


def agent_config_files(workspace: Path) -> tuple[Path, ...]:
    """The files in `<workspace>/.agent/` that decide what the agent may do on its own (rules, MCP servers), as the
    file system sees them (symlinks followed)."""
    folder = workspace / ".agent"
    return tuple((folder / name).resolve() for name in ("settings.json", "mcp.json"))


@dataclass(frozen=True)
class RuleFile:
    allow: tuple[Rule, ...]
    deny: tuple[Rule, ...]
    invalid: tuple[tuple[str, str], ...]  # (rule text, reason)
    raw: bytes | None  # the exact bytes that were parsed; None if the file is missing or could not be read
    error: str | None  # unreadable, bad JSON, wrong shape or a symlink: the file contributes no rules at all
    list_errors: tuple[str, ...] = ()  # an `allow` or `deny` that is not a list of strings: only that list is lost
    data: Mapping[str, object] | None = None  # the parsed top-level object (other keys, e.g. `guardrails`); None: no object


def _failed(error: str, raw: bytes | None = None) -> RuleFile:
    return RuleFile((), (), (), raw, error)


def _texts(value: object) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def read_rules(path: Path, *, follow_symlink: bool = False) -> RuleFile:
    """Read one settings file. A missing file is no rules and no error.

    A symlink (the file, or the folder it is in) is refused unless `follow_symlink`: a repository could point its
    own file anywhere.
    """
    if not follow_symlink and (path.is_symlink() or path.parent.is_symlink()):
        return _failed("is a symlink (or is in a folder that is one)")
    try:
        raw = read_regular(path)
        text = raw.decode("utf-8")
    except FileNotFoundError:
        return RuleFile((), (), (), None, None)
    except (OSError, UnicodeDecodeError) as exc:
        return _failed(f"cannot read: {exc}")
    try:
        data = parse_json(text)
    except ValueError as exc:
        return _failed(f"invalid JSON: {exc}", raw)
    if not isinstance(data, dict):
        return _failed("top level must be an object", raw)
    permissions = data.get("permissions", {})
    if not isinstance(permissions, dict):
        return dataclasses.replace(_failed('"permissions" must be an object', raw), data=data)
    invalid: list[tuple[str, str]] = []
    list_errors: list[str] = []
    parsed: dict[str, tuple[Rule, ...]] = {}
    for name in ("allow", "deny"):  # each list on its own: a bad `allow` must not switch off the `deny` rules
        texts = _texts(permissions.get(name))
        if texts is None:
            list_errors.append(f'"permissions.{name}" must be a list of strings')
            parsed[name] = ()
            continue
        rules: list[Rule] = []
        for text in texts:
            try:
                rules.append(parse_rule(text))
            except RuleError as exc:
                invalid.append((text, str(exc)))
        parsed[name] = tuple(rules)
    return RuleFile(parsed["allow"], parsed["deny"], tuple(invalid), raw, None, tuple(list_errors), data)


def _load_for_write(path: Path) -> tuple[dict[str, object], bytes | None, int]:
    """The settings file at `path` as `(top-level object, bytes read (None: missing), mode)` for a writer, with the
    guards every writer shares. Raises `OSError` for a symlink (the file or its folder), a non-regular, oversized or
    read-only file, and `ValueError` when it is not a JSON object: the user's file is never overwritten."""
    folder = path.parent
    if path.is_symlink() or folder.is_symlink():
        raise OSError(f"{path} is a symlink; not writing through it")
    data: dict[str, object] = {}
    mode = 0o600
    raw: bytes | None = None
    try:
        raw = read_regular(path)
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        pass
    else:
        if not os.access(path, os.W_OK):
            raise OSError(f"{path} is not writable; not changing it")
        try:
            loaded = parse_json(raw.decode("utf-8"))
        except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError
            raise ValueError(f"{path} is not valid JSON ({exc}); not changing it") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} is not a JSON object; not changing it")
        data = loaded
    return data, raw, mode


def _write_atomic(path: Path, data: dict[str, object], mode: int) -> bytes:
    """Write `data` as the settings file (a temp file in the same folder, then `os.replace`); returns the bytes."""
    folder = path.parent
    content = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".settings-", suffix=".tmp")  # mode 0600
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return content


def add_allow(path: Path, rule_text: str) -> tuple[bytes | None, bytes]:
    """Add `rule_text` to the allow list of the settings file at `path`; returns `(bytes read, new bytes)`: what
    the file held when this call read it (None: it did not exist) and what it holds now. Both come from this one
    read, so a caller can tell whether the content it just wrote is built on what it had seen before.

    Everything else in the file is kept, a rule already there is not added twice, and the write is atomic (a
    temp file in the same folder, then `os.replace`), so a crash leaves the old file or the new one. Raises
    `OSError` for a symlink (the file or its folder) or a failed write, and `ValueError` when an existing file
    is not a JSON object of the expected shape: the user's file is never overwritten.
    """
    data, raw, mode = _load_for_write(path)
    permissions = data.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError(f'"permissions" in {path} is not an object; not changing it')
    allow = permissions.setdefault("allow", [])
    if not isinstance(allow, list):
        raise ValueError(f'"permissions.allow" in {path} is not a list; not changing it')
    if rule_text in allow and raw is not None:
        return raw, raw  # nothing to add: the file stays byte for byte as it is
    allow.append(rule_text)
    return raw, _write_atomic(path, data, mode)


def set_top_key(path: Path, key: str, value: str) -> tuple[bytes | None, bytes]:
    """Set the top-level string `key` of the settings file at `path`; returns `(bytes read, new bytes)` like
    `add_allow`, with the same guards and the same atomic write. Every other key is kept; when the key already
    has that value the file stays byte for byte as it is."""
    data, raw, mode = _load_for_write(path)
    if raw is not None and data.get(key) == value:
        return raw, raw
    data[key] = value
    return raw, _write_atomic(path, data, mode)
