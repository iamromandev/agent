"""What a launch knows about permission rules: both files read once, the trust check, and the runtime `Rules`."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.engine.hooks.config import read_hooks
from src.engine.mcp.trust import TrustStore
from src.engine.permissions.config import RuleFile, add_allow, project_settings_path, read_rules, set_top_key
from src.engine.permissions.rules import Rule, Source, parse_rule, resolve_key
from src.engine.text import escape
from src.engine.tools.base import Tool

MAX_WARNINGS = 20  # more than this at launch is a wall of text: the rest is a count


class RuleSavedNotTrusted(OSError):
    """The rule is in the project file, but the file's new content could not be marked trusted."""


@dataclass(frozen=True)
class Verdict:
    rule: Rule
    source: Source


class Rules:
    """The rules a run decides with. Created once at launch and shared by the agent and its subagents:
    unlike `Grants` it is not reset by `/new` or a session switch."""

    def __init__(
        self,
        global_file: RuleFile,
        project_file: RuleFile,
        project_trusted: bool,
        save: Callable[[str], bytes | None] | None = None,
        trust: Callable[[bytes], None] | None = None,
        project_hooks: bool = False,
        save_key: Callable[[str, str], bytes | None] | None = None,
    ) -> None:
        """`save(rule_text)` writes the rule to the project file and returns the new bytes when they may be trusted
        (None: written, but the file was changed since launch, so its content is not vouched for). `trust(bytes)`
        remembers bytes as trusted. `project_hooks`: the project file holds hooks, which wait for the same trust.
        `save_key(key, value)` is `save` for a top-level string key (the model choice)."""
        self._global = global_file
        self._project = project_file
        self._trusted = project_trusted
        self._save = save
        self._save_key = save_key
        self._trust = trust
        self._project_hooks = project_hooks
        self._added: list[Rule] = []  # saved with "Always here" in this run: active at once
        self._deny: tuple[tuple[Rule, Source], ...] = (*((r, "global") for r in global_file.deny), *((r, "project") for r in project_file.deny))
        self._global_allow: tuple[Rule, ...] = global_file.allow

    @staticmethod
    def empty() -> Rules:
        none = RuleFile((), (), (), None, None)
        return Rules(none, none, True)

    @property
    def can_save(self) -> bool:
        return self._save is not None

    @property
    def untrusted_allow(self) -> tuple[Rule, ...]:
        """The project file's allow rules that wait for the user's trust (empty once trusted, or when it has none)."""
        return () if self._trusted else self._project.allow

    @property
    def project_trusted(self) -> bool:
        """Is the project file's content trusted (so its allow rules and hooks are active)?"""
        return self._trusted

    @property
    def pending_trust(self) -> bool:
        """Does the project file hold allow rules or hooks that wait for the user's trust?"""
        return not self._trusted and (bool(self._project.allow) or self._project_hooks)

    def deny(self, tool: str, args: Mapping[str, Any], key: str | None) -> Verdict | None:
        """A deny rule that matches, from either file. Project deny rules need no trust: they only refuse."""
        for rule, source in self._deny:
            if rule.denies(tool, args, key):
                return Verdict(rule, source)
        return None

    def allow(self, tool: str, args: Mapping[str, Any], key: str | None) -> Verdict | None:
        """An allow rule that matches: global ones, the project's only once trusted, and those saved in this run."""
        for rule in self._global_allow:
            if rule.matches(tool, args, key):
                return Verdict(rule, "global")
        project: tuple[Rule, ...] = (*(self._project.allow if self._trusted else ()), *self._added)
        for rule in project:
            if rule.matches(tool, args, key):
                return Verdict(rule, "project")
        return None

    def trust_project(self) -> None:
        """The user trusted the project file: its allow rules are active now. Raises `OSError` if the choice cannot
        be remembered; the rules stay active for this run."""
        self._trusted = True
        if self._trust is not None and self._project.raw is not None:
            self._trust(self._project.raw)

    def save_project(self, rule_text: str) -> None:
        """Add the exact rule to the project file ("Always here"); it applies at once.

        The new bytes become trusted only when the file's rules and hooks were already trusted (or it had none) and
        nobody changed it since launch: choosing "Always here" for one call must not also trust the rest of a file
        the user has not accepted. Raises `OSError` / `ValueError` when it cannot be saved (then nothing changes), and
        `RuleSavedNotTrusted` when it was saved but the trust store refused the new bytes (the rule is active now).
        """
        rule = parse_rule(rule_text)
        if self._save is None:
            raise OSError("there is no project settings file to save to")
        pending = self.pending_trust
        new = self._save(rule_text)
        if rule not in self._added and rule not in self._project.allow:
            self._added.append(rule)
        self._vouch(new, pending)

    def save_project_key(self, key: str, value: str) -> None:
        """Set a top-level string key of the project file (the model choice), under the same trust rules as
        `save_project`: a file that was trusted stays trusted, so the next launch asks nothing; a file the user
        did not trust, or one changed since launch, is not vouched for. Same errors as `save_project`."""
        if self._save_key is None:
            raise OSError("this setting cannot be saved here (these rules have no way to write the project file)")
        pending = self.pending_trust
        new = self._save_key(key, value)
        self._vouch(new, pending)

    def _vouch(self, new: bytes | None, pending: bool) -> None:
        """Trust the bytes just written when they may be: see `save_project`."""
        if new is not None and not pending and self._trust is not None:
            try:
                self._trust(new)
            except OSError as exc:
                raise RuleSavedNotTrusted(
                    f"saved to .agent/settings.json but could not be remembered as trusted ({exc}); "
                    "the next launch will ask"
                ) from exc
            self._trusted = True

    def listing(self) -> list[str]:
        """The lines `/permissions` shows; empty when there is nothing to show. Text that came from a file is escaped."""
        lines: list[str] = []
        for source, file in (("global", self._global), ("project", self._project)):
            trusted = source == "global" or self._trusted
            lines.extend(
                f"allow {escape(r.text)} ({source})" + ("" if trusted else " inactive: untrusted") for r in file.allow
            )
            if source == "project":
                lines.extend(f"allow {escape(r.text)} (project)" for r in self._added if r not in file.allow)
            lines.extend(f"deny {escape(r.text)} ({source})" for r in file.deny)
            lines.extend(f"invalid {escape(text)} ({source}): {escape(reason)}" for text, reason in file.invalid)
            lines.extend(f"error ({source}): {escape(error)}" for error in file.list_errors)
            if file.error is not None:
                lines.append(f"error ({source}): {escape(file.error)}")
        if not lines:
            return []
        if self._project.error is not None:
            state = "unusable"
        elif self.pending_trust:
            state = "untrusted"
        else:
            state = "trusted" if (self._project.raw is not None or self._added) else "none"
        return [*lines, f"project file: {state}"]


@dataclass(frozen=True)
class PermissionSetup:
    rules: Rules
    warnings: tuple[str, ...]
    global_data: Mapping[str, object] | None = None  # each file's parsed top-level object, for the other settings keys
    project_data: Mapping[str, object] | None = None


_URL_HOST = re.compile(r"[a-z][a-z0-9+.-]*://[^/]*$", re.IGNORECASE)


def tool_keys(tools: Iterable[Tool]) -> dict[str, str | None]:
    """`{tool name: its key argument}` for `prepare`: what a valued rule for that tool is compared with."""
    return {tool.name: resolve_key(tool.name, tool.grant_arg) for tool in tools}


def _rule_warnings(path: Path, file: RuleFile, *, project: bool, keys: Mapping[str, str | None] | None) -> list[str]:
    """Warnings for one file. The text of a rule, an error and the path all come from outside: escaped by the caller."""
    warnings = [f"{path}: {text}: {reason}" for text, reason in file.invalid]
    warnings.extend(f"{path}: {error}; that list adds no rules" for error in file.list_errors)
    if file.error is not None:
        tail = "; its deny rules are not applied either" if project else "; it adds no rules"
        warnings.append(f"{path}: {file.error}{tail}")
    for rule in (*file.allow, *file.deny):
        if rule.value is not None and _keyless(rule.tool, keys):
            warnings.append(f"{path}: {rule.text}: {rule.tool} has no argument to match; this rule never applies")
    for rule in file.allow:
        # A prefix is a plain string prefix: `https://docs.python.org*` also matches `https://docs.python.org.evil.com`.
        if rule.tool == "web_fetch" and rule.prefix and rule.value is not None and _URL_HOST.match(rule.value):
            warnings.append(f"{path}: {rule.text}: a web_fetch prefix should end the host with a / (it also matches longer host names)")
    return warnings


def _keyless(tool: str, keys: Mapping[str, str | None] | None) -> bool:
    """Is `tool` known to have no key argument? An MCP tool never has one; others only when `keys` says so."""
    if tool.startswith("mcp__"):
        return keys is not None
    return keys is not None and tool in keys and keys[tool] is None


def prepare(
    workspace: Path, global_path: Path, store: TrustStore, keys: Mapping[str, str | None] | None = None
) -> PermissionSetup:
    """Read the global and the project settings file once; the project's bytes are hashed as they were parsed.

    `keys` (see `tool_keys`) lets it warn about a rule that names a tool's argument when the tool has none."""
    project_path = project_settings_path(workspace)
    global_file = read_rules(global_path, follow_symlink=True)  # the user's own file: dotfile setups link it
    project_file = read_rules(project_path)
    raw = project_file.raw
    project_hooks = bool(read_hooks(project_file.data, "project")[0])  # valid hooks run commands: they wait for the same trust
    needs_trust = (bool(project_file.allow) or project_hooks) and raw is not None and not store.is_trusted(workspace, raw)
    warnings = [
        escape(w)
        for w in (
            *_rule_warnings(global_path, global_file, project=False, keys=keys),
            *_rule_warnings(project_path, project_file, project=True, keys=keys),
        )
    ]
    if needs_trust and store.inside(workspace):  # a repository's own "trust": never honoured, never written
        warnings.append(
            escape(f"permissions: the trust store {store.path} is inside the workspace, so project allow rules and hooks stay untrusted")
        )
    seen = {"raw": raw}  # what the file held when it was last read or written here

    def save(rule_text: str) -> bytes | None:
        read, new = add_allow(project_path, rule_text)  # what it read and what it wrote come from one read
        unchanged = read == seen["raw"]
        if unchanged:
            seen["raw"] = new
        return new if unchanged else None

    def save_key(key: str, value: str) -> bytes | None:
        read, new = set_top_key(project_path, key, value)
        unchanged = read == seen["raw"]
        if unchanged:
            seen["raw"] = new
        return new if unchanged else None

    def trust(content: bytes) -> None:
        store.trust(workspace, content)

    rules = Rules(global_file, project_file, not needs_trust, save, trust, project_hooks, save_key)
    if len(warnings) > MAX_WARNINGS:
        more = len(warnings) - MAX_WARNINGS
        warnings = [*warnings[:MAX_WARNINGS], f"and {more} more warnings"]
    return PermissionSetup(rules, tuple(warnings), global_file.data, project_file.data)
