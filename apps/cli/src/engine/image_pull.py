"""`/images`: the state of the two image models in the Hugging Face cache, and a background warm-up that downloads them.

A warm-up is one tiny mflux job per model (256x256, 1 step, no timeout) through the same `MfluxGenerator` as the tools,
so the filtered environment and the process group are the same. UI-free: no `textual`, no `httpx`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import struct
import tempfile
import time
import zlib
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.engine.env import child_environment
from src.engine.imagegen import (
    KNOWN_MODELS,
    ImageGenError,
    ImageRequest,
    MfluxGenerator,
    hub_cache,
    is_alias,
    png_size,
    resolve_model,
    run_with_ticker,
    snapshot_files,
)
from src.engine.process import kill_group
from src.engine.text import escape, escape_display

if TYPE_CHECKING:
    from src.engine.guardrails.setup import Guardrails

Role = Literal["generate", "edit", "upscale"]
ROLES: tuple[Role, ...] = ("generate", "edit", "upscale")
MARGIN = 2_000_000_000
SIZE_TIMEOUT = 60.0
WARMUP_SIDE, WARMUP_STEPS, WARMUP_SEED = 256, 1, 0
WARMUP_PROMPT = "a plain grey square"
UPSCALE_WARMUP_SIDE = 64  # the source of the upscale warm-up; at 2x the output is 128x128
HF_COMMAND = "hf"
NOT_INSTALLED = "image tools are not installed (uv sync --directory cli --extra image on Apple Silicon)"
REFUSED = "a model warm-up is running (/images cancel stops it)"
_OUTPUT_CAP = 1024 * 1024
_UNITS = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000, "T": 1_000_000_000_000}

State = Literal["cached", "partial", "missing", "local folder", "unavailable"]


@dataclass(frozen=True)
class ModelStatus:
    role: Role
    model: str
    state: State
    size: int | None


class PullSizeError(Exception):
    """The download size could not be read; the text is the reason."""


def format_gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def format_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    minutes, secs = divmod(whole, 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def repo_folder(model: str, env: Mapping[str, str] = os.environ) -> Path:
    repo = resolve_model(model)[0]  # an alias lives in its repo
    return hub_cache(env) / f"models--{repo.replace('/', '--')}"


def model_status(role: Role, model: str | None, env: Mapping[str, str] = os.environ) -> ModelStatus:
    """What is on disk for `model`: stat calls only, so a 15 GB cache answers at once."""
    if model is None:
        return ModelStatus(role, "", "unavailable", None)
    if not is_alias(model) and Path(model).is_dir():  # like mflux: a directory relative to the cwd wins over a repo id such as `org/name`
        return ModelStatus(role, model, "local folder", None)
    repo, files = resolve_model(model)
    if files is not None:  # a known alias: only the files mflux fetches count (the repo holds much more)
        found, unfinished = snapshot_files(repo, files, env, KNOWN_MODELS[model.lower()].revision)
        size = sum(found.values())
        if len(found) == len(files) and not unfinished:
            return ModelStatus(role, model, "cached", size)
        if found or unfinished:
            return ModelStatus(role, model, "partial", size)
        return ModelStatus(role, model, "missing", None)
    folder = repo_folder(model, env)
    complete = False
    snapshot_seen = False
    try:
        for entry in (folder / "snapshots").iterdir():
            if entry.is_dir():
                snapshot_seen = True
                complete = complete or any(entry.iterdir())
    except OSError:
        pass
    size = 0
    incomplete = False
    blobs_seen = False
    try:
        for blob in (folder / "blobs").iterdir():
            blobs_seen = True
            incomplete = incomplete or blob.name.endswith(".incomplete")
            with contextlib.suppress(OSError):
                size += blob.stat().st_size
    except OSError:
        pass
    if complete and not incomplete:
        return ModelStatus(role, model, "cached", size)
    if incomplete or snapshot_seen:
        return ModelStatus(role, model, "partial", size)
    return ModelStatus(role, model, "missing", size if blobs_seen else None)


def format_status(rows: list[ModelStatus], pulling: str | None) -> str:
    lines = []
    for row in rows:
        if row.state == "unavailable":
            lines.append(f"{row.role}: unavailable (mflux command not found)")
            continue
        size = f", {format_gb(row.size)}" if row.size else ""
        lines.append(f"{row.role}: {escape_display(row.model)} - {row.state}{size}")
    if pulling is not None:
        lines.append(f"pulling {escape_display(pulling)}")
    return "\n".join(lines)


def parse_dry_run(stdout: str, files: tuple[str, ...] | None = None) -> int:
    """Bytes the download would fetch: the rows of `hf download --dry-run` inside a subfolder (the diffusers layout
    mflux reads), without the cached ones (`-`). Root files such as example images are not read by mflux. With `files`
    (a known model) the rows are exactly those paths instead, whatever the folder."""
    lines = stdout.splitlines()
    try:
        start = lines.index("file\tsize")
    except ValueError:
        raise PullSizeError("no file table in the hf output") from None
    total = 0.0
    for line in lines[start + 1 :]:
        name, tab, size = line.partition("\t")
        if not tab or not size or not name or size == "-":
            continue
        if (name not in files) if files is not None else ("/" not in name):
            continue
        number, unit = size[:-1], size[-1]
        if unit.isdigit():
            number, unit = size, ""
        try:
            total += float(number) * _UNITS[unit]
        except (ValueError, KeyError):
            raise PullSizeError("unreadable size in the hf output") from None
    return round(total)


def free_space(folder: Path) -> tuple[int, Path]:
    """Free bytes on the volume of `folder`, asked on its nearest existing parent."""
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free, probe


def grey_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + b"\x80" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


async def download_size(
    hf: Path, model: str, guard: Guardrails, on_start: Callable[[int], None] | None = None
) -> int:
    """Bytes `hf download <repo> --dry-run` says are still missing (see `parse_dry_run`); a known alias is asked as
    its repo id and counted by its files."""
    repo, files = resolve_model(model)
    try:
        proc = await asyncio.create_subprocess_exec(
            str(hf), "download", repo, "--dry-run",
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=DEVNULL,
            start_new_session=True,
            env=guard.env.filter(child_environment()),
        )
    except OSError as exc:
        raise PullSizeError(f"could not start hf: {escape(str(exc))}") from exc
    if on_start is not None:
        on_start(proc.pid)
    kept = bytearray()

    async def drain() -> None:
        assert proc.stdout is not None
        while chunk := await proc.stdout.read(65_536):
            if len(kept) < _OUTPUT_CAP:
                kept.extend(chunk)
        await proc.wait()

    try:
        await asyncio.wait_for(drain(), SIZE_TIMEOUT)
    except TimeoutError:
        await kill_group(proc)
        raise PullSizeError("timed out") from None
    except BaseException:
        await kill_group(proc)
        raise
    if proc.returncode != 0:
        raise PullSizeError(f"hf exited {proc.returncode}")
    return parse_dry_run(bytes(kept).decode("utf-8", errors="replace"), files)


Level = Literal["info", "warn"]
Emit = Callable[[Level, str], Awaitable[None]]


class ImagePuller:
    """Warms the image models up in the background. One pull at a time; the caller runs `pull` in its own task."""

    def __init__(
        self,
        generators: Mapping[Role, MfluxGenerator | None],
        hf: Path | None,
        guard: Guardrails,
        env: Mapping[str, str] = os.environ,
        disk: Callable[[Path], tuple[int, Path]] = free_space,
    ) -> None:
        self.generators = generators
        self.hf = hf
        self.guard = guard
        self.env = env
        self.disk = disk
        self.busy = False
        self.pulling: str | None = None
        self._reserved = False
        self._task: asyncio.Task[None] | None = None
        self._pgid: int | None = None

    def reserve(self) -> None:
        """Claim the pull before its task starts, so `busy` is true at once; `pull` takes the claim."""
        if self.busy:
            raise RuntimeError("an image pull is already running")
        self.busy = True
        self._reserved = True

    def release(self) -> None:
        """Give back a reservation that `pull` never took."""
        if self._reserved:
            self._reserved = False
            self.busy = False

    @property
    def installed(self) -> bool:
        return any(g is not None for g in self.generators.values())

    def status(self) -> list[ModelStatus]:
        rows: list[ModelStatus] = []
        for role in ROLES:
            generator = self.generators.get(role)
            rows.append(model_status(role, None if generator is None else generator.model, self.env))
        return rows

    async def pull(self, roles: tuple[Role, ...], emit: Emit, progress: Callable[[str], None]) -> None:
        if self._reserved:
            self._reserved = False
        elif self.busy:
            raise RuntimeError("an image pull is already running")
        self.busy = True
        self._task = asyncio.current_task()
        try:
            for role in roles:
                await self._pull_one(role, emit, progress)
        except asyncio.CancelledError:
            await emit("warn", "image pull cancelled")
            raise
        finally:
            self.busy = False
            self.pulling = None
            self._task = None
            self._pgid = None
            progress("")

    async def _pull_one(self, role: Role, emit: Emit, progress: Callable[[str], None]) -> None:
        generator = self.generators.get(role)
        if generator is None:
            await emit("warn", f"{role} model pull failed: mflux command not found")
            return
        model = generator.model
        shown = escape_display(model)
        status = model_status(role, model, self.env)
        note = ""
        if status.state != "local folder":
            try:
                if self.hf is None:
                    raise PullSizeError("hf not found")
                try:
                    needed = await download_size(self.hf, model, self.guard, self._started_pid)
                finally:
                    self._pgid = None
                free, folder = self.disk(hub_cache(self.env))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = str(exc) if isinstance(exc, PullSizeError) else type(exc).__name__
                await emit("warn", f"could not read the download size of {shown} ({escape_display(reason)}); pulling anyway")
            else:
                if needed == 0:
                    pass
                elif free < needed + MARGIN:
                    await emit(
                        "warn",
                        f"not enough disk space for {shown}: needs {format_gb(needed)} + 2 GB, "
                        f"{format_gb(free)} free in {escape_display(str(folder))}",
                    )
                    return
                else:
                    note = f" (download size {format_gb(needed)}, {format_gb(free)} free)"
        await emit("info", f"pulling {role} model {shown}{note}")
        started = time.monotonic()
        out_dir = Path(tempfile.mkdtemp(prefix="agent-image-"))
        self.pulling = f"{role}: starting…"

        def sink(text: str) -> None:
            self.pulling = f"{role}: {text}"
            progress(self.pulling)

        try:
            inputs: tuple[Path, ...] = ()
            if role in ("edit", "upscale"):
                side = UPSCALE_WARMUP_SIDE if role == "upscale" else WARMUP_SIDE
                source = out_dir / "in-0.png"
                fd = os.open(source, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(grey_png(side, side))
                inputs = (source,)
            if role == "upscale":
                request = ImageRequest(
                    "", 2 * UPSCALE_WARMUP_SIDE, 2 * UPSCALE_WARMUP_SIDE, WARMUP_STEPS, WARMUP_SEED, inputs, scale=2
                )
            else:
                request = ImageRequest(WARMUP_PROMPT, WARMUP_SIDE, WARMUP_SIDE, WARMUP_STEPS, WARMUP_SEED, inputs)
            warm = MfluxGenerator(
                generator.command, generator.model, None, generator.guard, extra=generator.extra, kind=generator.kind
            )
            output = await run_with_ticker(warm, request, out_dir, sink, on_start=self._started_pid)
            if png_size(output.read_bytes()) is None:
                raise ImageGenError("mflux wrote no image")
        except (ImageGenError, OSError) as exc:
            reason = str(exc) if isinstance(exc, ImageGenError) else (exc.strerror or str(exc))
            await emit("warn", f"{role} model pull failed: {escape_display(reason)}")
            return
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            self._pgid = None
            progress("")
        after = model_status(role, model, self.env)
        size = f", {format_gb(after.size)}" if after.size else ""
        await emit("info", f"{role} model ready ({shown}{size}, {format_elapsed(time.monotonic() - started)})")

    def _started_pid(self, pid: int) -> None:
        self._pgid = pid

    async def cancel(self) -> bool:
        """Stop a running pull (its process group dies with it); False when none was running."""
        task = self._task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.wait({task})
        return True

    def kill_all(self) -> None:
        pgid = self._pgid
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
