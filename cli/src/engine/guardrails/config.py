"""The `guardrails` key of the settings files, read from their already parsed JSON (see `RuleFile.data`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.text import escape


@dataclass(frozen=True)
class GuardConfig:
    sandbox_enabled: bool = True
    network: bool = True
    extra_write: tuple[str, ...] = ()  # global only
    extra_protected: tuple[str, ...] = ()  # global + project
    keep_env: frozenset[str] = frozenset()  # global only, upper-cased


_ONLY_GLOBAL = "can only be set in the global settings; ignored"


class _Reader:
    """Collects values and warnings for one file. `where` is the escaped file path."""

    def __init__(self, where: str, warnings: list[str]) -> None:
        self.where = where
        self.warnings = warnings

    def warn(self, text: str) -> None:
        self.warnings.append(f"{self.where}: {text}")

    def section(self, data: Mapping[str, object] | None, key: str, label: str) -> Mapping[str, object]:
        """`data[key]` when it is an object; `{}` when it is absent or has the wrong type (that warns)."""
        value = data.get(key) if data is not None else None
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.warn(f'"{label}" must be an object; ignored')
            return {}
        return value

    def boolean(self, section: Mapping[str, object], key: str, label: str) -> bool | None:
        value = section.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            self.warn(f'"{label}" must be true or false; ignored')
            return None
        return value

    def texts(self, section: Mapping[str, object], key: str, label: str) -> tuple[str, ...] | None:
        value = section.get(key)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            self.warn(f'"{label}" must be a list of non-empty strings; ignored')
            return None
        return tuple(value)

    def patterns(self, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """`protected` patterns, trimmed, without the ones that cannot work: an empty one, a `..` segment (the paths
        that are matched never have one) or a bare `~` (that is a file name, not the home folder). A `/*` ending is
        kept with a warning: it covers only the direct children."""
        if values is None:
            return None
        kept: list[str] = []
        for raw in values:
            pattern = raw.strip()
            if not pattern:
                self.warn("guardrails.protected pattern is empty after trimming spaces; ignored")
            elif pattern == "~":
                self.warn('guardrails.protected pattern "~" matches only a file named "~"; ignored (use "~/**" for the home folder)')
            elif ".." in pattern.split("/"):
                self.warn(f'guardrails.protected pattern "{escape(pattern)}" has a ".." segment; ignored')
            else:
                if pattern.endswith("/*"):
                    self.warn(
                        f'guardrails.protected pattern "{escape(pattern)}" covers only the direct children of the folder, '
                        f'not what is deeper; use "{escape(pattern)}*" for everything inside'
                    )
                kept.append(pattern)
        return tuple(kept)


def read_guardrails(
    global_data: Mapping[str, object] | None,
    project_data: Mapping[str, object] | None,
    *,
    global_path: Path | str = "global settings",
    project_path: Path | str = ".agent/settings.json",
) -> tuple[GuardConfig, list[str]]:
    """The effective guardrails settings and the launch warnings (escaped).

    The global file may set everything. The project file (a repository's own) may only tighten: add `protected`
    patterns and set `sandbox.network` to false; the rest is ignored with a warning.
    """
    warnings: list[str] = []

    glob = _Reader(escape(str(global_path)), warnings)
    g_guard = glob.section(global_data, "guardrails", "guardrails")
    g_sandbox = glob.section(g_guard, "sandbox", "guardrails.sandbox")
    enabled = glob.boolean(g_sandbox, "enabled", "guardrails.sandbox.enabled")
    network = glob.boolean(g_sandbox, "network", "guardrails.sandbox.network")
    write = glob.texts(g_sandbox, "write", "guardrails.sandbox.write")
    protected = glob.patterns(glob.texts(g_guard, "protected", "guardrails.protected"))
    keep = glob.texts(g_guard, "keepEnv", "guardrails.keepEnv")

    proj = _Reader(escape(str(project_path)), warnings)
    p_guard = proj.section(project_data, "guardrails", "guardrails")
    p_sandbox = proj.section(p_guard, "sandbox", "guardrails.sandbox")
    p_enabled = proj.boolean(p_sandbox, "enabled", "guardrails.sandbox.enabled")
    p_network = proj.boolean(p_sandbox, "network", "guardrails.sandbox.network")
    p_write = proj.texts(p_sandbox, "write", "guardrails.sandbox.write")
    p_protected = proj.patterns(proj.texts(p_guard, "protected", "guardrails.protected"))
    p_keep = proj.texts(p_guard, "keepEnv", "guardrails.keepEnv")
    if p_enabled is False:
        proj.warn(f"guardrails.sandbox.enabled {_ONLY_GLOBAL}")
    if p_network is True:
        proj.warn(f"guardrails.sandbox.network {_ONLY_GLOBAL}")
    if p_write:
        proj.warn(f"guardrails.sandbox.write {_ONLY_GLOBAL}")
    if p_keep:
        proj.warn(f"guardrails.keepEnv {_ONLY_GLOBAL}")

    config = GuardConfig(
        sandbox_enabled=True if enabled is None else enabled,
        network=False if p_network is False else (True if network is None else network),
        extra_write=write or (),
        extra_protected=(protected or ()) + (p_protected or ()),
        keep_env=frozenset(name.upper() for name in keep or ()),
    )
    return config, warnings
