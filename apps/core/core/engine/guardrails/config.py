"""The `guardrails` key of the settings files, read from their already parsed JSON (see `RuleFile.data`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from core.engine.text import escape


@dataclass(frozen=True)
class GuardConfig:
    extra_protected: tuple[str, ...] = ()  # global + project
    keep_env: frozenset[str] = frozenset()  # global only, upper-cased


_ONLY_GLOBAL = "can only be set in the global settings; ignored"
REMOVED_KEY_WARNING = "guardrails.sandbox is no longer supported and is ignored"  # the macOS sandbox was removed


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

    def check_sandbox(self, section: Mapping[str, object]) -> None:
        """One warning when the file still has the removed `sandbox` key; its value is never read."""
        if "sandbox" in section:
            self.warn(REMOVED_KEY_WARNING)

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

    The global file may set everything. The project file (a repository's own) may only add `protected` patterns.
    A `sandbox` key (the removed sandbox's settings) in either file is ignored with one warning, whatever its value.
    """
    warnings: list[str] = []

    glob = _Reader(escape(str(global_path)), warnings)
    g_guard = glob.section(global_data, "guardrails", "guardrails")
    glob.check_sandbox(g_guard)
    protected = glob.patterns(glob.texts(g_guard, "protected", "guardrails.protected"))
    keep = glob.texts(g_guard, "keepEnv", "guardrails.keepEnv")

    proj = _Reader(escape(str(project_path)), warnings)
    p_guard = proj.section(project_data, "guardrails", "guardrails")
    proj.check_sandbox(p_guard)
    p_protected = proj.patterns(proj.texts(p_guard, "protected", "guardrails.protected"))
    p_keep = proj.texts(p_guard, "keepEnv", "guardrails.keepEnv")
    if p_keep:
        proj.warn(f"guardrails.keepEnv {_ONLY_GLOBAL}")

    config = GuardConfig(
        extra_protected=(protected or ()) + (p_protected or ()),
        keep_env=frozenset(name.upper() for name in keep or ()),
    )
    return config, warnings
