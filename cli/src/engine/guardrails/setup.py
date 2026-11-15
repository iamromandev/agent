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
from src.engine.guardrails.redact import Redactor
from src.engine.guardrails.sandbox import Sandbox

if TYPE_CHECKING:
    from src.engine.tools.base import ToolContext, ToolResult

STUB_SANDBOX_REASON = "not built yet"


@dataclass
class Guardrails:
    paths: ProtectedPaths
    env: EnvFilter
    sandbox: Sandbox
    redactor: Redactor
    warnings: tuple[str, ...] = ()
    removed_env: tuple[str, ...] = ()  # names only, never values
    redactions: int = 0  # how many secrets were replaced in this run, for /guardrails

    def redact_result(self, result: ToolResult) -> ToolResult:
        """`result` with secrets replaced in its output and its display copy."""
        output, count = self.redactor.redact(result.output)
        display = result.display
        if display is not None:
            display, more = self.redactor.redact(display)
            count += more
        self.redactions += count
        if count == 0:
            return result
        return dataclasses.replace(result, output=output, display=display)

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
) -> Guardrails:
    env_filter = EnvFilter(config.keep_env)
    sandbox = Sandbox(None, STUB_SANDBOX_REASON, config.network, ())
    return Guardrails(
        ProtectedPaths(home, config.extra_protected, workspace, also_home=also_home),
        env_filter,
        sandbox,
        Redactor(env_filter.secret_values(env)),
        warnings,
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
) -> Guardrails:
    """Build the launch's `Guardrails` from both settings files' parsed data and the agent's own environment.

    `system` (`platform.system()`) and `probe` are for the sandbox, which a later phase adds."""
    config, warnings = read_guardrails(global_data, project_data, global_path=global_path, project_path=project_path)
    # A `home` given by the caller is used as it is; otherwise both `Path.home()` (from `$HOME`) and the account's
    # recorded home count, because `$HOME` can name any folder.
    return _build(
        config,
        home if home is not None else Path.home(),
        env,
        tuple(warnings),
        workspace,
        login_home() if home is None else None,
    )


def guard_of(ctx: ToolContext) -> Guardrails:
    """The context's guard; the built-in defaults when it has none (tests, bare tool use)."""
    return ctx.guard if ctx.guard is not None else Guardrails.off(ctx.workspace)
