"""What a launch knows about permission rules: both files read once, the trust check, and the runtime `Rules`."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.engine.configwatch import TrustRevoked
from core.engine.hooks.config import read_hooks
from core.engine.lsp.config import project_needs_trust
from core.engine.mcp.trust import TrustStore
from core.engine.permissions.config import RuleFile, add_allow, project_settings_path, read_rules, set_top_key
from core.engine.permissions.rules import Rule, Source, parse_rule, resolve_key
from core.engine.text import escape
from core.engine.tools.base import Tool

MAX_WARNINGS = 20  # more than this at launch is a wall of text: the rest is a count


class RuleSavedNotTrusted(OSError):
    """The rule is in the project file, but the file's new content could not be marked trusted."""


OwnWrite = Callable[[Sequence[Path]], AbstractContextManager[None]]


def _no_watch(paths: Sequence[Path]) -> AbstractContextManager[None]:
    return nullcontext()


@dataclass(frozen=True)
class Verdict:
    rule: Rule
    source: Source | Literal["run"]  # "run": a rule given for this run only (`--allow`)


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
        *,
        settings_path: Path | None = None,
        store_path: Path | None = None,
        own_write: OwnWrite | None = None,
        project_lsp: bool = False,
    ) -> None:
        """`save(rule_text)` writes the rule to the project file and returns the new bytes when they may be trusted
        (None: written, but the file was changed since launch, so its content is not vouched for). `trust(bytes)`
        remembers bytes as trusted. `project_hooks`: the project file holds hooks, which wait for the same trust;
        `project_lsp`: so does an `lsp` value that starts language servers.
        `save_key(key, value)` is `save` for a top-level string key (the model choice). `own_write(paths)` wraps
        each of those writes (the config watch's `own_write`: it accepts only the paths named); none: no watch."""
        self._global = global_file
        self._project = project_file
        self._trusted = project_trusted
        self._run_trusted = False  # trusted by `trust_for_run` only: the trust store is never written
        self._save = save
        self._save_key = save_key
        self._trust = trust
        self._project_hooks = project_hooks
        self._project_lsp = project_lsp
        self._added: list[Rule] = []  # saved with "Always here" in this run: active at once
        self._run: list[Rule] = []  # given for this run only (`--allow`): never saved
        self._deny: tuple[tuple[Rule, Source], ...] = (*((r, "global") for r in global_file.deny), *((r, "project") for r in project_file.deny))
        self._global_allow: tuple[Rule, ...] = global_file.allow
        self._global_active = True
        self._global_revoked = False  # switched off by a change seen in this run: `enable_global` refuses
        self._revoked = False  # the config watch saw the agent's config change: no project trust for the rest of the run
        self._lock = threading.Lock()  # the revoke may come from another thread (a trust write runs in one)
        # Around a write to watched files, given the paths it writes (the config watch accepts only those).
        self._own_write: OwnWrite = own_write if own_write is not None else _no_watch
        self.settings_path = settings_path  # the project settings file `save` and `save_key` write
        self.store_path = store_path  # the trust store `trust` writes

    @property
    def own_write(self) -> OwnWrite:
        """What wraps the rules' own writes; given to the constructor (or `prepare`), never set afterwards (read-only, so
        a late assignment fails loudly instead of leaving the writes unwatched)."""
        return self._own_write

    @staticmethod
    def empty() -> Rules:
        none = RuleFile((), (), (), None, None)
        return Rules(none, none, True)

    @property
    def can_save(self) -> bool:
        return self._save is not None and not self._revoked

    @property
    def global_active(self) -> bool:
        """Are the global file's allow rules (and, through `Hooks`, its hooks) in force? Deny rules never depend on it."""
        return self._global_active

    def revoke_project(self) -> None:
        """The agent's config was changed by a child process: the project file is untrusted for the rest of the run.
        Its allow rules, the rules added with "Always here" in this run and a trust given for this run only all stop;
        nothing can trust the project again or save to its file until the next launch. Deny rules still apply."""
        with self._lock:
            self._revoked = True
            self._trusted = False
            self._run_trusted = False
            self._added.clear()

    @property
    def revoked(self) -> bool:
        """Was project trust revoked in this run (see `revoke_project`)?"""
        return self._revoked

    def _paths(self, *paths: Path | None) -> list[Path]:
        return [p for p in paths if p is not None]

    def _set_trusted(self) -> None:
        """Turn the project's trust on, unless a revoke came first (then nothing changes and `TrustRevoked`)."""
        with self._lock:
            if self._revoked:
                raise TrustRevoked("trusting the project")
            self._trusted = True

    def disable_global(self, *, until_trusted: bool = False) -> None:
        """The global config was changed by a child process: its allow rules stop for the rest of the run.
        `until_trusted` (the launch marker of an earlier run's change): they stop until `enable_global`."""
        with self._lock:
            self._global_active = False
            if not until_trusted:
                self._global_revoked = True

    def enable_global(self) -> None:
        """The user trusted the global config at launch: its allow rules (and hooks) are active again. Raises
        `TrustRevoked`, and changes nothing, once the config watch saw a change in this run (project or global)."""
        with self._lock:
            if self._revoked or self._global_revoked:
                raise TrustRevoked("trusting the global config", reason="the agent's config was changed by a command in this run")
            self._global_active = True

    @property
    def global_inactive(self) -> str | None:
        """Why the global allow rules and hooks are off, as `/permissions` and `/hooks` say it; None when they are on."""
        if self._global_active:
            return None
        return "inactive: global config changed" if self._global_revoked else "inactive: held until trusted"

    @property
    def global_revoked(self) -> bool:
        """Did a change seen in this run switch the global rules off (so `enable_global` refuses)?"""
        return self._global_revoked

    @property
    def global_allow(self) -> tuple[Rule, ...]:
        """The global file's allow rules (also while they are off)."""
        return self._global_allow

    @property
    def global_deny(self) -> tuple[Rule, ...]:
        """The global file's deny rules: they always apply (deny only tightens)."""
        return tuple(rule for rule, source in self._deny if source == "global")

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
        """Does the project file hold allow rules, hooks or language servers that wait for the user's trust?"""
        return not self._trusted and (bool(self._project.allow) or self._project_hooks or self._project_lsp)

    def deny(self, tool: str, args: Mapping[str, Any], key: str | None) -> Verdict | None:
        """A deny rule that matches, from either file. Project deny rules need no trust: they only refuse."""
        for rule, source in self._deny:
            if rule.denies(tool, args, key):
                return Verdict(rule, source)
        return None

    def allow(self, tool: str, args: Mapping[str, Any], key: str | None) -> Verdict | None:
        """An allow rule that matches: global ones, the project's only once trusted, and those saved in this run."""
        for rule in self._global_allow if self._global_active else ():
            if rule.matches(tool, args, key):
                return Verdict(rule, "global")
        project: tuple[Rule, ...] = (*(self._project.allow if self._trusted else ()), *self._added)
        for rule in project:
            if rule.matches(tool, args, key):
                return Verdict(rule, "project")
        for rule in self._run:
            if rule.matches(tool, args, key):
                return Verdict(rule, "run")
        return None

    def add_run_allow(self, rules: Iterable[Rule]) -> None:
        """Allow these rules for this run only: checked after the settings files' rules, never saved or remembered,
        and deny rules still win."""
        self._run.extend(rules)

    def trust_for_run(self) -> None:
        """Treat the project file as trusted for this run only: nothing is written to the trust store."""
        with self._lock:
            if self._revoked:
                return  # a revoke during the run wins over a trust given at launch or late
            self._trusted = True
            self._run_trusted = True

    def trust_project(self) -> None:
        """The user trusted the project file: its allow rules are active now. Raises `OSError` if the choice cannot
        be remembered; the rules stay active for this run. Raises `TrustRevoked` (and trusts nothing) once project
        trust was revoked in this run, also when the check that opens the store write is what revoked it."""
        self._set_trusted()
        if self._trust is not None and self._project.raw is not None:
            with self._own_write(self._paths(self.store_path)):
                if self._revoked:
                    raise TrustRevoked("trusting the project")
                self._trust(self._project.raw)
        if self._revoked:  # the check after the write saw a change: the guard already forgot the store entry
            raise TrustRevoked("trusting the project")

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
        if self._revoked:
            raise TrustRevoked("saving the rule")
        pending = self.pending_trust
        with self._own_write(self._paths(self.settings_path, self.store_path)):
            if self._revoked:  # the check that opens the own write saw a change: nothing is written
                raise TrustRevoked("saving the rule")
            new = self._save(rule_text)
        with self._lock:
            if self._revoked:  # the check after the write saw a change: the rule is not active
                raise TrustRevoked("saving the rule")
            if rule not in self._added and rule not in self._project.allow:
                self._added.append(rule)
        self._vouch(new, pending)

    def save_project_key(self, key: str, value: str) -> None:
        """Set a top-level string key of the project file (the model choice), under the same trust rules as
        `save_project`: a file that was trusted stays trusted, so the next launch asks nothing; a file the user
        did not trust, or one changed since launch, is not vouched for. Same errors as `save_project`."""
        if self._save_key is None:
            raise OSError("this setting cannot be saved here (these rules have no way to write the project file)")
        if self._revoked:
            raise TrustRevoked("saving the setting")
        pending = self.pending_trust
        with self._own_write(self._paths(self.settings_path)):
            if self._revoked:
                raise TrustRevoked("saving the setting")
            new = self._save_key(key, value)
        if self._revoked:
            raise TrustRevoked("saving the setting")
        self._vouch(new, pending)

    def _vouch(self, new: bytes | None, pending: bool) -> None:
        """Trust the bytes just written when they may be: see `save_project`. Never after a revoke."""
        if new is not None and not pending and self._trust is not None and not self._run_trusted:
            try:
                with self._own_write(self._paths(self.store_path)):
                    if self._revoked:
                        raise TrustRevoked("trusting the saved file")
                    self._trust(new)
            except TrustRevoked:
                raise
            except OSError as exc:
                raise RuleSavedNotTrusted(
                    f"saved to .agent/settings.json but could not be remembered as trusted ({exc}); "
                    "the next launch will ask"
                ) from exc
            self._set_trusted()

    def listing(self) -> list[str]:
        """The lines `/permissions` shows; empty when there is nothing to show. Text that came from a file is escaped."""
        lines: list[str] = []
        for source, file in (("global", self._global), ("project", self._project)):
            off = self.global_inactive if source == "global" else (None if self._trusted else "inactive: untrusted")
            lines.extend(f"allow {escape(r.text)} ({source})" + (f" {off}" if off else "") for r in file.allow)
            if source == "project":
                lines.extend(f"allow {escape(r.text)} (project)" for r in self._added if r not in file.allow)
            lines.extend(f"deny {escape(r.text)} ({source})" for r in file.deny)
            lines.extend(f"invalid {escape(text)} ({source}): {escape(reason)}" for text, reason in file.invalid)
            lines.extend(f"error ({source}): {escape(error)}" for error in file.list_errors)
            if file.error is not None:
                lines.append(f"error ({source}): {escape(file.error)}")
        lines.extend(f"allow {escape(r.text)} (run)" for r in self._run)
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


def run_rule_warnings(rules: Iterable[Rule], keys: Mapping[str, str | None], offered: Iterable[str] = ()) -> list[str]:
    """Warnings for rules given for one run (`--allow`): a tool that is not offered (`keys` has the built-in tools,
    `offered` more names, e.g. MCP tools) or a valued rule on a tool without a key argument. Never applies, so say so."""
    names = {*keys, *offered}
    warnings: list[str] = []
    for rule in rules:
        if rule.server_wildcard or rule.tool.startswith("mcp__"):
            continue  # the servers start later: their tool names are not known yet
        if rule.tool not in names:
            warnings.append(f"--allow {rule.text}: no tool named {rule.tool}; this rule never applies")
        elif rule.value is not None and _keyless(rule.tool, keys):
            warnings.append(f"--allow {rule.text}: {rule.tool} has no argument to match; this rule never applies")
    return [escape(w) for w in warnings]


def _keyless(tool: str, keys: Mapping[str, str | None] | None) -> bool:
    """Is `tool` known to have no key argument? An MCP tool never has one; others only when `keys` says so."""
    if tool.startswith("mcp__"):
        return keys is not None
    return keys is not None and tool in keys and keys[tool] is None


def prepare(
    workspace: Path,
    global_path: Path,
    store: TrustStore,
    keys: Mapping[str, str | None] | None = None,
    *,
    own_write: OwnWrite | None = None,
) -> PermissionSetup:
    """Read the global and the project settings file once; the project's bytes are hashed as they were parsed.

    `keys` (see `tool_keys`) lets it warn about a rule that names a tool's argument when the tool has none.
    `own_write` (the config watch's) wraps the rules' own writes to the project file and the trust store."""
    project_path = project_settings_path(workspace)
    global_file = read_rules(global_path, follow_symlink=True)  # the user's own file: dotfile setups link it
    project_file = read_rules(project_path)
    raw = project_file.raw
    project_hooks = bool(read_hooks(project_file.data, "project")[0])  # valid hooks run commands: they wait for the same trust
    project_lsp = project_needs_trust(global_file.data, project_file.data, workspace)  # language servers are programs the file starts
    needs_trust = (bool(project_file.allow) or project_hooks or project_lsp) and raw is not None and not store.is_trusted(workspace, raw)
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

    rules = Rules(
        global_file,
        project_file,
        not needs_trust,
        save,
        trust,
        project_hooks,
        save_key,
        settings_path=project_path,
        store_path=store.path,
        own_write=own_write,
        project_lsp=project_lsp,
    )
    if len(warnings) > MAX_WARNINGS:
        more = len(warnings) - MAX_WARNINGS
        warnings = [*warnings[:MAX_WARNINGS], f"and {more} more warnings"]
    return PermissionSetup(rules, tuple(warnings), global_file.data, project_file.data)
