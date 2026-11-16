"""What a launch knows about guardrails: the settings read once, and the `Guardrails` value tools and spawn sites share."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.engine.guardrails.config import GuardConfig, read_guardrails
from src.engine.guardrails.env import EnvFilter
from src.engine.guardrails.paths import ProtectedPaths, login_home
from src.engine.guardrails.redact import Redactor, drop_partial_tail
from src.engine.guardrails.sandbox import Sandbox, default_probe, make_sandbox

if TYPE_CHECKING:
    from src.engine.tools.base import Preview, ToolContext, ToolResult

REDACTION_FAILED = "[redaction failed: output withheld]"
NO_SANDBOX_REASON = "off: not configured"  # `Guardrails.off`: tests and a context without a guard
PREVIEW_WINDOW = 100_000  # what redaction reads of a diff preview: the dialog cannot show more, and the cost is bounded


@dataclass
class Guardrails:
    paths: ProtectedPaths
    env: EnvFilter
    sandbox: Sandbox
    redactor: Redactor
    warnings: tuple[str, ...] = ()
    info: tuple[str, ...] = ()  # launch lines that are not warnings (the sandbox switched off in the settings)
    removed_env: tuple[str, ...] = ()  # names only, never values
    redactions: int = 0  # how many secrets were replaced in this run, for /guardrails

    def redact_result(self, result: ToolResult) -> ToolResult:
        """`result` with secrets replaced in its output and its display copy.

        Only the output is counted: the display is the same text with colours, and counting it too would count one
        secret twice. When redaction changed anything the display copy is dropped, so the transcript shows the redacted
        output: a secret split by colour codes (`ghp_`, a reset, the rest) is in no piece of the copy that a pattern
        can match. Fails closed (NFR4): if redaction raises, the output is withheld and nothing of it is kept. The
        error text is not used: it could quote the text being redacted."""
        from src.engine.tools.base import ToolResult  # function level: the tools package imports this module

        try:
            output, count = self.redactor.redact(result.output)
            display = result.display
            if display is not None:
                display = self.redactor.redact(display)[0]
        except Exception:
            return ToolResult(REDACTION_FAILED, True, "redaction failed")
        self.redactions += count
        if output == result.output and display == result.display:
            return result
        return dataclasses.replace(result, output=output, display=None)

    def redact_preview(self, preview: Preview | None) -> Preview | None:
        """The preview of a call as the user sees it: the body of a diff holds lines of the file as it is, so secrets
        in it are replaced. A command or the full content of a write is the model's own text and shown as it is (the
        user approves exactly what runs). Not counted, and never what a tool writes: only the display changes. A diff
        longer than `PREVIEW_WINDOW` is cut there (the last word with it) and ends with a note of how much is not shown."""
        if preview is None or preview.kind != "diff":
            return preview
        body = preview.body
        hidden = max(len(body) - PREVIEW_WINDOW, 0)
        try:
            shown = self.redactor.redact(body[:PREVIEW_WINDOW])[0]
            if hidden:
                shown, dropped = drop_partial_tail(shown)
                shown += f"\n[{hidden + dropped} characters not shown]"
        except Exception:
            shown = REDACTION_FAILED
        return preview if shown == body else dataclasses.replace(preview, body=shown)

    @classmethod
    def off(cls, workspace: Path) -> Guardrails:
        """For tests and a `ToolContext` without a guard: built-in protected paths, the filter on, no sandbox, no
        scan of the workspace. Built once per home folder and process (the scan of the protected folders is not free,
        and every tool call without a guard asks for one); each caller gets its own redaction counter."""
        return dataclasses.replace(_off(Path.home()))


@functools.lru_cache(maxsize=8)
def _off(home: Path) -> Guardrails:
    return _build(GuardConfig(), home, {}, also_home=login_home())


def _build(
    config: GuardConfig,
    home: Path,
    env: Mapping[str, str],
    warnings: tuple[str, ...] = (),
    workspace: Path | None = None,
    also_home: Path | None = None,
    sandbox_for: Callable[[ProtectedPaths], tuple[Sandbox, str | None]] | None = None,
) -> Guardrails:
    env_filter = EnvFilter(config.keep_env)
    paths = ProtectedPaths(home, config.extra_protected, workspace, also_home=also_home)
    info: tuple[str, ...] = ()
    if sandbox_for is None:
        sandbox = Sandbox(None, NO_SANDBOX_REASON, config.network, ())
    else:
        sandbox, warning = sandbox_for(paths)
        if warning is not None:
            warnings = (*warnings, *warning.split("\n"))
        if not config.sandbox_enabled:
            info = (f"sandbox: {sandbox.reason}",)
    return Guardrails(
        paths,
        env_filter,
        sandbox,
        Redactor(env_filter.secret_values(env)),
        warnings,
        info,
        tuple(env_filter.removed(env)),
    )


def prepare(
    workspace: Path,
    global_data: Mapping[str, object] | None,
    project_data: Mapping[str, object] | None,
    env: Mapping[str, str],
    system: str,
    probe: Callable[[list[str]], bool] | None = None,
    *,
    home: Path | None = None,
    global_path: Path | str = "global settings",
    project_path: Path | str = ".agent/settings.json",
    sessions: Path | None = None,
) -> Guardrails:
    """Build the launch's `Guardrails` from both settings files' parsed data and the agent's own environment.

    `system` (`platform.system()`) and `probe` (runs the sandboxed trivial command; the real one by default) decide
    whether the sandbox is on; `sessions` is the sessions folder, which the sandbox never lets a command write."""
    config, warnings = read_guardrails(global_data, project_data, global_path=global_path, project_path=project_path)
    # A `home` given by the caller is used as it is; otherwise both `Path.home()` (from `$HOME`) and the account's
    # recorded home count, because `$HOME` can name any folder.
    home_folder = home if home is not None else Path.home()

    def sandbox_for(paths: ProtectedPaths) -> tuple[Sandbox, str | None]:
        return make_sandbox(
            config, workspace, home_folder, env, system, probe or default_probe, sessions=sessions, protected=paths
        )

    return _build(config, home_folder, env, tuple(warnings), workspace, login_home() if home is None else None, sandbox_for)


def guard_of(ctx: ToolContext) -> Guardrails:
    """The context's guard; the built-in defaults when it has none (tests, bare tool use)."""
    return ctx.guard if ctx.guard is not None else Guardrails.off(ctx.workspace)
