"""The speak tool: narrate English text into a WAV in the workspace with a local Kokoro model, after approval."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from src.engine.imagegen import PROGRESS_TICK, ImageGenError, _tail, model_cached, run_helper
from src.engine.media import (
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    MAX_AUDIO_BYTES,
    MAX_SPEED,
    MAX_TEXT,
    MIN_SPEED,
    SKIPPED_NAME,
    TEXT_NAME,
    VOICES,
    WAV_NAME,
    wav_info,
)
from src.engine.media import (
    SPEECH_HELPER as _FOUND_HELPER,
)
from src.engine.provider.images import size_text
from src.engine.text import escape_display

from .base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from .outputs import TooBig, check_output_path, finish_placement, place_bytes, read_nofollow

SPEECH_HELPER: Path | None = _FOUND_HELPER  # a module constant so tests can swap in a fake helper
MAX_SKIPPED_SHOWN = 20
_SKIPPED_READ = 64 * 1024


def _invalid(message: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary="invalid")


def _failed(message: str) -> ToolResult:
    return ToolResult(message, is_error=True, summary="failed")


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def _skipped_words(path: Path) -> list[str]:
    """The words the helper could not pronounce; empty when the file is missing or not a list of strings."""
    try:
        raw = read_nofollow(path, _SKIPPED_READ)
        words = json.loads(raw)
    except (OSError, TooBig, ValueError):
        return []
    if not isinstance(words, list):
        return []
    found: list[str] = []
    for word in words:
        if isinstance(word, str) and word.strip() and word not in found:
            found.append(word)
    return found


def _skipped_text(words: list[str]) -> str:
    if not words:
        return ""
    shown = ", ".join(escape_display(word[:40]) for word in words[:MAX_SKIPPED_SHOWN])
    more = len(words) - MAX_SKIPPED_SHOWN
    return f"; skipped words: {shown}" + (f" and {more} more" if more > 0 else "")


class SpeakTool(Tool):
    name: ClassVar[str] = "speak"
    description: ClassVar[str] = (
        "Narrate English text into a WAV file with a local text-to-speech model (Kokoro) and save it at a path "
        "inside the workspace. Words the model does not know are left out and listed in the result; respell them "
        "phonetically and call again. It takes seconds to a minute, replaces a file at the path, and every call "
        "needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": f"What to say, English, 1-{MAX_TEXT} characters."},
            "path": {"type": "string", "description": "Where to save it: a .wav path inside the workspace."},
            "voice": {
                "type": "string",
                "enum": list(VOICES),
                "description": f"A voice; af_/am_ American, bf_/bm_ British. Default {DEFAULT_VOICE}.",
            },
            "speed": {
                "type": "number",
                "minimum": MIN_SPEED,
                "maximum": MAX_SPEED,
                "description": f"Speaking speed, {MIN_SPEED}-{MAX_SPEED}. Default {DEFAULT_SPEED}.",
            },
        },
        "required": ["text", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "path"

    def __init__(self, model: str, timeout: float, guard: Any) -> None:
        self.model = model
        self.timeout = timeout
        self.guard = guard

    def label(self, args: dict[str, Any]) -> str:
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        raw: str = args["path"]
        text: str = args["text"].strip()
        voice = args.get("voice", DEFAULT_VOICE)
        speed = args.get("speed", DEFAULT_SPEED)
        if not 1 <= len(text) <= MAX_TEXT:
            return _invalid(f"text must be 1-{MAX_TEXT} characters")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return _invalid("text is not valid Unicode")
        if voice not in VOICES:
            return _invalid(f"voice must be one of {', '.join(VOICES)}")
        if isinstance(speed, bool) or not isinstance(speed, int | float) or not MIN_SPEED <= speed <= MAX_SPEED:
            return _invalid(f"speed must be {MIN_SPEED}-{MAX_SPEED}")
        if isinstance(checked := check_output_path(raw, ctx, ".wav"), ToolResult):
            return checked
        path = checked
        lines = [
            f"text: {escape_display(text)}",
            f"voice: {voice}, speed {speed:g}",
            f"model: {escape_display(self.model)}",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        if not model_cached(self.model):
            lines.append(
                f"[model {escape_display(self.model)} is not downloaded; this call may download several hundred MB]"
            )
        preview = Preview("content", f"speak {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await self._run(text, voice, float(speed), path, raw, ctx)

        return Prepared(preview, execute)

    async def _run(self, text: str, voice: str, speed: float, path: Path, raw: str, ctx: ToolContext) -> ToolResult:
        """Run the helper in a private folder, check the WAV it wrote and place it at `path`. The folder is always
        removed."""
        helper = SPEECH_HELPER
        if helper is None or not helper.is_file():
            return _failed("the speech helper is missing")
        folder = Path(tempfile.mkdtemp(prefix="agent-media-"))
        try:
            try:
                await finish_placement(asyncio.to_thread(_write_text, folder / TEXT_NAME, text))
            except OSError as exc:
                return _failed(f"Cannot stage the text: {exc.strerror or exc}")
            argv = [
                sys.executable, "-I", str(helper),
                "--model", self.model,
                "--voice", voice,
                "--speed", f"{speed:g}",
                "--text-file", TEXT_NAME,
                "--out", WAV_NAME,
                "--skipped", SKIPPED_NAME,
            ]  # fmt: skip
            try:
                code, stderr = await self._helper(argv, folder, ctx)
            except ImageGenError as exc:
                message = str(exc)
                return ToolResult(
                    message, is_error=True, summary="timed out" if message.startswith("timed out") else "failed"
                )
            if code != 0:
                return _failed(f"kokoro failed (exit {code}): {_tail(stderr)}")
            try:
                data = await asyncio.to_thread(read_nofollow, folder / WAV_NAME, MAX_AUDIO_BYTES)
            except TooBig:
                return _failed("kokoro wrote audio over 50 MB")
            except (FileNotFoundError, NotADirectoryError):
                return _failed("kokoro wrote no audio")
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK):  # a symlink where the audio should be
                    return _failed("kokoro wrote no audio")
                return _failed(f"Cannot read the audio: {exc.strerror or exc}")
            info = wav_info(data)
            if info is None:
                return _failed("kokoro wrote something that is not a WAV file")
            skipped = await asyncio.to_thread(_skipped_words, folder / SKIPPED_NAME)
            try:
                refused = await finish_placement(asyncio.to_thread(place_bytes, path, data, ctx))
            except OSError as exc:
                return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
            if refused is not None:
                return refused
            described = f"WAV, {info.seconds:.1f} s, {info.rate / 1000:g} kHz, {size_text(len(data))}"
            return ToolResult(f"spoke {raw} ({described}{_skipped_text(skipped)})", summary=described)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    async def _helper(self, argv: list[str], folder: Path, ctx: ToolContext) -> tuple[int, bytes]:
        """Run the helper; with a progress sink a ticker sends `speaking… m:ss` when it changes."""
        sink = ctx.progress
        if sink is None:
            return await run_helper(argv, folder, self.guard, self.timeout, who="the speech helper")
        from src.engine.events import ToolProgress  # not at module level: events imports the tools package

        start = time.monotonic()

        async def ticker() -> None:
            last = None
            while True:
                text = f"speaking… {_clock(time.monotonic() - start)}"
                if text != last:
                    last = text
                    with contextlib.suppress(Exception):
                        sink(ToolProgress(text))
                await asyncio.sleep(PROGRESS_TICK)

        task = asyncio.create_task(ticker())
        try:
            return await run_helper(argv, folder, self.guard, self.timeout, who="the speech helper")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _write_text(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
