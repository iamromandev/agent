"""Formatters of a run: chosen by file extension, one process per file and formatter, no shell."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import signal
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.engine.format.config import PLACEHOLDER, FormatSetup, FormatterSpec
from core.engine.guardrails.setup import Guardrails
from core.engine.imagegen import ImageGenError, _tail, run_helper
from core.engine.lsp.config import changed_since_launch
from core.engine.text import escape
from core.engine.undo import read_nofollow

FILE_LIMIT = 2 * 1024 * 1024
NOTE_LIMIT = 200


@dataclass(frozen=True)
class FormatOutcome:
    name: str
    changed_lines: int  # 0 when nothing changed or the run failed
    error: str | None = None


def count_changed(old: bytes, new: bytes) -> int:
    """+/- lines of a unified diff (at least 1 when the bytes differ at all, e.g. a final newline only)."""
    if old == new:
        return 0
    a = old.decode("utf-8", "replace").splitlines()
    b = new.decode("utf-8", "replace").splitlines()
    lines = sum(1 for x in difflib.unified_diff(a, b, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))
    return max(1, lines)


def _restore(path: Path, data: bytes) -> None:
    """Put `data` back as the file's content, atomically: a temporary file beside it (`O_EXCL|O_NOFOLLOW`, the same
    mode) renamed over it, so a reader never sees half a file and a symlink put there is replaced, not followed. Only
    when the content differs."""
    with contextlib.suppress(OSError):
        if read_nofollow(path, FILE_LIMIT + 1) == data:
            return
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    temp = path.with_name(f".{path.name}.fmt-{os.getpid()}-{os.urandom(4).hex()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        try:
            view = memoryview(data)
            while view:  # a short write is not an error: write the rest
                view = view[os.write(fd, view) :]
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


class FormatHub:
    def __init__(
        self,
        setup: FormatSetup,
        workspace: Path,
        guard: Guardrails,
        *,
        timeout: float,
        project_active: Callable[[], bool] = lambda: True,
        global_active: Callable[[], bool] = lambda: True,
    ) -> None:
        self._setup = setup
        self._workspace = workspace
        self._guard = guard
        self._timeout = timeout
        self._project_active = project_active
        self._global_active = global_active
        self._withdrawn: dict[str, str] = {}
        self._live: set[int] = set()  # process groups of running formatters

    def _trusted(self, spec: FormatterSpec) -> bool:
        return self._project_active() if spec.source == "project" else self._global_active()

    def _active(self, spec: FormatterSpec) -> bool:
        return spec.name not in self._withdrawn and self._trusted(spec)

    def _eligible(self, path: Path, written: bytes) -> bool:
        """R8: never a protected path, a target outside the workspace, a symlink, a hard-linked file, or a file that is
        no longer what the tool wrote."""
        try:
            if os.path.islink(path) or not path.resolve().is_relative_to(self._workspace.resolve()):
                return False
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1 or len(written) > FILE_LIMIT:
                return False
            if self._guard.paths.is_protected(path):
                return False
            return read_nofollow(path, FILE_LIMIT) == written
        except (OSError, RuntimeError):
            return False

    def _argv(self, spec: FormatterSpec, path: Path) -> list[str]:
        argv = [item.replace(PLACEHOLDER, str(path)) for item in spec.argv]
        return argv if any(PLACEHOLDER in item for item in spec.argv) else [*argv, str(path)]

    async def format(self, path: Path, written: bytes) -> list[FormatOutcome]:
        """Run the matching formatters on `path`, whose content is `written`. Never raises but for a cancel (the file is
        then put back as it was written)."""
        suffix = path.suffix.lower()
        specs = [s for s in self._setup.formatters if suffix in s.extensions and self._active(s)]
        if not specs or not await asyncio.to_thread(self._eligible, path, written):
            return []
        outcomes: list[FormatOutcome] = []
        current = written
        for spec in specs:
            if not self._active(spec):
                continue
            outcome, current = await self._one(spec, path, current)
            outcomes.append(outcome)
        return outcomes

    async def _one(self, spec: FormatterSpec, path: Path, before: bytes) -> tuple[FormatOutcome, bytes]:
        if spec.fingerprint is not None and changed_since_launch(spec.argv[0], spec.fingerprint):
            return FormatOutcome(spec.name, 0, "the program changed since launch; not run"), before
        failure: str | None = None
        started: list[int] = []

        def on_start(pid: int) -> None:
            started.append(pid)
            self._live.add(pid)

        try:
            code, stderr = await run_helper(
                self._argv(spec, path),
                self._workspace,
                self._guard,
                self._timeout,
                on_start=on_start,
                who=f"the formatter {spec.name}",
                env_extra=spec.env,
            )
            if code != 0:
                failure = f"exit {code}: {_tail(stderr)}"
        except ImageGenError as exc:
            failure = str(exc)
        except BaseException:
            with contextlib.suppress(OSError):
                _restore(path, before)  # Esc: the formatter was killed mid-write
            raise
        finally:
            self._live.difference_update(started)
        after = before
        if failure is None:
            try:
                after = read_nofollow(path, FILE_LIMIT)
            except OSError as exc:
                failure = f"cannot read the result: {escape(str(exc))}"
            else:
                if not after and before:
                    failure = "left an empty file"
        if failure is not None:
            try:
                _restore(path, before)
            except OSError as exc:
                failure += f" (could not restore: {escape(str(exc))})"
            return FormatOutcome(spec.name, 0, escape(failure)[:NOTE_LIMIT]), before
        return FormatOutcome(spec.name, count_changed(before, after)), after

    def listing(self) -> list[str]:
        """One line per table entry: `<name> (<source>) <found|not found|off[: reason]>  <extensions>`."""
        rows: list[str] = []
        for spec in self._setup.formatters:
            exts = " ".join(sorted(spec.extensions))
            if spec.name in self._withdrawn:
                state = f"off: {self._withdrawn[spec.name]}"
            elif not self._trusted(spec):
                state = "off: untrusted"
            else:
                state = "found"
            rows.append(f"{escape(spec.name)} ({spec.source}) {state}  {exts}")
        rows.extend(f"{escape(name)} (builtin) not found" for name in self._setup.missing)
        return rows

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._setup.warnings

    def sources(self) -> dict[str, str]:
        return {spec.name: spec.source for spec in self._setup.formatters}

    def untrusted(self, source: str = "project") -> list[FormatterSpec]:
        return [s for s in self._setup.formatters if s.source == source and s.name not in self._withdrawn and not self._trusted(s)]

    def withdraw(self, names: Sequence[str], reason: str) -> None:
        """Never run these again in this run."""
        for name in names:
            self._withdrawn[name] = reason

    def kill_all(self) -> None:
        for pgid in list(self._live):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
