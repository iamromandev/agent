"""The media tools: speak (narrate English text into a WAV with a local Kokoro model) and assemble_video (scenes of
images, narration and music into a vertical MP4 with ffmpeg), each after approval."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from src.engine.image_inputs import InputImage, resolve_input, stage_one
from src.engine.imagegen import PROGRESS_TICK, ImageGenError, _tail, model_cached, run_helper
from src.engine.media import (
    DEFAULT_CROSSFADE,
    DEFAULT_MOTION,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    FRAME,
    MAX_AUDIO_BYTES,
    MAX_CAPTION,
    MAX_CROSSFADE,
    MAX_IMAGE_BYTES,
    MAX_INPUT_BYTES,
    MAX_PIXELS,
    MAX_SCENE_S,
    MAX_SCENES,
    MAX_SPEED,
    MAX_TEXT,
    MAX_TOTAL_S,
    MAX_VIDEO_BYTES,
    MIN_SCENE_S,
    MIN_SPEED,
    MOTIONS,
    OUTPUT_NAME,
    SKIPPED_NAME,
    TEXT_NAME,
    VOICES,
    WAV_NAME,
    InputAudio,
    final_argv,
    format_text,
    mp4_info,
    plan_timeline,
    resolve_audio,
    scene_argv,
    wav_info,
)
from src.engine.media import (
    MEDIA_HELPER as _FOUND_MEDIA_HELPER,
)
from src.engine.media import (
    SPEECH_HELPER as _FOUND_HELPER,
)
from src.engine.provider.images import size_text
from src.engine.text import escape_display

from .base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from .outputs import TooBig, check_output_path, finish_placement, place_bytes, place_file, read_nofollow

SPEECH_HELPER: Path | None = _FOUND_HELPER  # a module constant so tests can swap in a fake helper
MEDIA_HELPER: Path | None = _FOUND_MEDIA_HELPER  # the same for assemble_video's frame and caption helper
MAX_SKIPPED_SHOWN = 20
_SKIPPED_READ = 64 * 1024


HEADER_BYTES = 4 * 1024 * 1024  # the part of the output that is read: moov is at the front (faststart)


def _read_header(path: Path, limit: int) -> tuple[bytes, int]:
    """The first `HEADER_BYTES` of a regular file opened without following a symlink, and its size from the same
    descriptor (`TooBig` past `limit`)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise FileNotFoundError(path)
        if info.st_size > limit:
            raise TooBig
        return handle.read(HEADER_BYTES), info.st_size


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


def _write_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def _is_regular(path: Path) -> bool:
    """A regular file that is not a symlink: what a step wrote into the private folder."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


@dataclasses.dataclass(frozen=True)
class _Scene:
    image: InputImage
    audio: InputAudio | None
    length: float
    motion: str
    text: str  # the raw text, for the dialog
    caption: str  # wrapped, '' for none


@dataclasses.dataclass(frozen=True)
class _Plan:
    scenes: tuple[_Scene, ...]
    music: InputAudio | None
    crossfade: float


class _Stop(Exception):
    """A step ended the call early; `result` is what the call returns."""

    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.output)
        self.result = result


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class AssembleVideoTool(Tool):
    name: ClassVar[str] = "assemble_video"
    description: ClassVar[str] = (
        f"Assemble a vertical {FRAME[0]}x{FRAME[1]} MP4 from scenes: each scene is an image (a workspace path or "
        "clipboard:N) with a slow pan or zoom, optional narration (a workspace WAV, for example from speak: the scene "
        "lasts as long as it) or a duration, and an optional caption; scenes are crossfaded, music (a WAV, MP3 or "
        "M4A) is mixed under the narration and fades out. Captions use Arial Bold or Helvetica, so scripts they lack "
        "(CJK, Bengali, Devanagari) show as boxes. Saves an .mp4 inside the workspace, replacing a file at the "
        f"path. It takes about a minute for a short reel and every call needs user approval. At most {MAX_SCENES} "
        f"scenes, {MAX_SCENE_S} s each and {MAX_TOTAL_S} s in all."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SCENES,
                "description": f"1-{MAX_SCENES} scenes, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "A workspace image path or clipboard:N."},
                        "audio": {"type": "string", "description": "A workspace WAV path: the narration of the scene."},
                        "duration": {
                            "type": "number",
                            "minimum": MIN_SCENE_S,
                            "maximum": MAX_SCENE_S,
                            "description": "Seconds; required without audio, ignored with it.",
                        },
                        "motion": {"type": "string", "enum": list(MOTIONS), "description": f"Default {DEFAULT_MOTION}."},
                        "text": {
                            "type": "string",
                            "description": f"A caption, at most {MAX_CAPTION} characters (about 4 short lines).",
                        },
                    },
                    "required": ["image"],
                },
            },
            "path": {"type": "string", "description": "Where to save it: an .mp4 path inside the workspace."},
            "music": {"type": "string", "description": "A workspace WAV, MP3 or M4A path for the background music."},
            "crossfade": {
                "type": "number",
                "minimum": 0,
                "maximum": MAX_CROSSFADE,
                "description": f"Seconds of crossfade between scenes, 0-{MAX_CROSSFADE}. Default {DEFAULT_CROSSFADE}.",
            },
        },
        "required": ["scenes", "path"],
    }
    mutating: ClassVar[bool] = True  # hidden and refused in plan mode
    needs_approval: ClassVar[bool] = True  # every call asks; target stays None so accept-edits never skips it
    uncancellable: ClassVar[bool | None] = False  # Esc kills the process group
    grant_arg: ClassVar[str | None] = "path"

    def __init__(self, ffmpeg: Path, timeout: float, guard: Any) -> None:
        self.ffmpeg = ffmpeg
        self.timeout = timeout
        self.guard = guard

    def label(self, args: dict[str, Any]) -> str:
        path = args.get("path")
        return f"{self.name} {path}" if isinstance(path, str) and path else self.name

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if (problem := check_args(self.parameters, args)) is not None:
            return _invalid(problem)
        raw: str = args["path"]
        crossfade = args.get("crossfade", DEFAULT_CROSSFADE)
        if not _number(crossfade) or not 0 <= crossfade <= MAX_CROSSFADE:
            return _invalid(f"crossfade must be 0-{MAX_CROSSFADE:.1f}")
        music_ref = args.get("music")
        if music_ref is not None and not isinstance(music_ref, str):
            return _invalid('"music" must be a string')
        if isinstance(checked := self._check_scenes(args["scenes"]), ToolResult):
            return checked
        entries = checked
        if isinstance(path := check_output_path(raw, ctx, ".mp4"), ToolResult):
            return path
        scenes: list[_Scene] = []
        spent = 0  # bytes staged so far: an image or narration once per scene that uses it, the music once

        def over_budget() -> ToolResult | None:
            if spent <= MAX_INPUT_BYTES:
                return None
            return _invalid(f"the inputs total {spent / (1024 * 1024):.0f} MB; at most {MAX_INPUT_BYTES // (1024 * 1024)} MB")

        for number, (ref, audio_ref, duration, motion, text, caption) in enumerate(entries, 1):
            image = await asyncio.to_thread(resolve_input, ref, ctx, MAX_IMAGE_BYTES)
            if isinstance(image, ToolResult):
                return image
            spent += len(image.data)
            if (refused := over_budget()) is not None:
                return refused
            if image.width * image.height > MAX_PIXELS:
                return _invalid(f"scene {number} image {image.label} is {image.width}\N{MULTIPLICATION SIGN}{image.height}: over 50 megapixels")
            audio = None
            length = duration
            if audio_ref is not None:
                found = await asyncio.to_thread(resolve_audio, audio_ref, ctx, ("wav",))
                if isinstance(found, ToolResult):
                    return found
                info = wav_info(found.data)
                if info is None:
                    return _invalid(f"scene {number} audio {found.label} is not a PCM WAV file with sound")
                audio, length = found, info.seconds
                spent += len(found.data)
                if (refused := over_budget()) is not None:
                    return refused
            assert length is not None
            if length > MAX_SCENE_S:
                return _invalid(f"scene {number} is {length:.1f} s; at most {MAX_SCENE_S} s")
            if length < MIN_SCENE_S:
                return _invalid(f"scene {number} is {length:.1f} s; at least {MIN_SCENE_S:g} s")
            scenes.append(_Scene(image, audio, length, motion, text, caption))
        music = None
        if music_ref is not None:
            found = await asyncio.to_thread(resolve_audio, music_ref, ctx, ("wav", "mp3", "m4a"))
            if isinstance(found, ToolResult):
                return found
            music = found
            spent += len(found.data)
            if (refused := over_budget()) is not None:
                return refused
        timeline = plan_timeline([s.length for s in scenes], float(crossfade))
        if timeline.total > MAX_TOTAL_S:
            return _invalid(f"the video would be {timeline.total:.1f} s; at most {MAX_TOTAL_S} s")
        plan = _Plan(tuple(scenes), music, float(crossfade))
        lines = []
        for number, scene in enumerate(scenes, 1):
            line = f"scene {number}: {scene.image.label}, "
            if scene.audio is not None:
                line += f"narration {scene.audio.label} ({scene.length:.1f} s)"
            else:
                line += f"{scene.length:.1f} s"
            line += f", {scene.motion}"
            if scene.text:
                line += f", text: {escape_display(scene.text[:60])}"
            lines.append(line)
        lines += [
            f"music: {music.label if music is not None else 'none'}",
            f"crossfade: {crossfade:g} s",
            f"total: {timeline.total:.1f} s",
        ]
        if path.exists():
            lines.append("[overwrites the existing file]")
        preview = Preview("content", f"assemble_video {escape_display(raw)}", "\n".join(lines))

        async def execute() -> ToolResult:
            return await self._run(plan, path, raw, ctx)

        return Prepared(preview, execute)

    @staticmethod
    def _check_scenes(
        scenes: Any,
    ) -> list[tuple[str, str | None, float | None, str, str, str]] | ToolResult:
        """The scenes' own fields, checked without touching a file: (image, audio, duration, motion, text, caption)."""
        if not isinstance(scenes, list):
            return _invalid('"scenes" must be an array')
        if not 1 <= len(scenes) <= MAX_SCENES:
            return _invalid(f"scenes must have 1-{MAX_SCENES} items")
        entries: list[tuple[str, str | None, float | None, str, str, str]] = []
        for number, scene in enumerate(scenes, 1):
            if not isinstance(scene, dict):
                return _invalid(f"scene {number} must be an object")
            image, audio = scene.get("image"), scene.get("audio")
            if not isinstance(image, str) or not image:
                return _invalid(f"scene {number} needs an image")
            if audio is not None and not isinstance(audio, str):
                return _invalid(f"scene {number} audio must be a string")
            duration = scene.get("duration")
            if audio is None and duration is None:
                return _invalid(f"scene {number} needs audio or a duration")
            if duration is not None and (not _number(duration) or not MIN_SCENE_S <= duration <= MAX_SCENE_S):
                return _invalid(f"scene {number} duration must be {MIN_SCENE_S:g}-{MAX_SCENE_S} s")
            motion = scene.get("motion", DEFAULT_MOTION)
            if motion not in MOTIONS:
                return _invalid(f"scene {number} motion must be one of {', '.join(MOTIONS)}")
            text = scene.get("text", "")
            if not isinstance(text, str):
                return _invalid(f"scene {number} text must be a string")
            text = text.strip()
            if len(text) > MAX_CAPTION:
                return _invalid(f"scene {number} text must be at most {MAX_CAPTION} characters")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                return _invalid(f"scene {number} text is not valid Unicode")
            caption = format_text(text)
            if caption is None:
                return _invalid(f"scene {number} text needs more than 4 lines; shorten it")
            entries.append((image, audio, None if duration is None else float(duration), motion, text, caption))
        return entries

    def _say(self, ctx: ToolContext, text: str) -> None:
        sink = ctx.progress
        if sink is None:
            return
        from src.engine.events import ToolProgress  # not at module level: events imports the tools package

        with contextlib.suppress(Exception):
            sink(ToolProgress(text))

    async def _step(self, argv: list[str], folder: Path, deadline: float, who: str) -> tuple[int, bytes]:
        """One process under the call's single deadline; the whole group dies on timeout or cancel."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise _Stop(ToolResult(f"timed out after {self.timeout:g} s", is_error=True, summary="timed out"))
        try:
            return await run_helper(argv, folder, self.guard, left, who=who)
        except ImageGenError as exc:
            if str(exc).startswith("timed out"):
                raise _Stop(
                    ToolResult(f"timed out after {self.timeout:g} s", is_error=True, summary="timed out")
                ) from None
            raise _Stop(_failed(str(exc))) from None

    async def _run(self, plan: _Plan, path: Path, raw: str, ctx: ToolContext) -> ToolResult:
        """Stage the inputs in a private folder, make the frames and captions, render each scene, join them, check the
        MP4 and place it at `path`. One deadline covers every process. The folder is always removed."""
        helper = MEDIA_HELPER
        if helper is None or not helper.is_file():
            return _failed("the media helper is missing")
        folder = Path(tempfile.mkdtemp(prefix="agent-media-"))
        deadline = time.monotonic() + self.timeout
        try:
            try:
                return await self._assemble(plan, path, raw, ctx, helper, folder, deadline)
            except _Stop as stop:
                return stop.result
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    async def _assemble(
        self, plan: _Plan, path: Path, raw: str, ctx: ToolContext, helper: Path, folder: Path, deadline: float
    ) -> ToolResult:
        scenes = plan.scenes
        count = len(scenes)
        try:
            staged = await finish_placement(asyncio.to_thread(self._stage, plan, folder))
        except OSError as exc:
            return _failed(f"Cannot stage the inputs: {exc.strerror or exc}")
        music_name, inputs, narrations = staged
        base = [sys.executable, "-I", str(helper)]
        for i, scene in enumerate(scenes):
            self._say(ctx, f"framing {i + 1}/{count}")
            argv = [*base, "frame", "--in", inputs[i], "--out", f"frame-{i}.png", "--width", str(FRAME[0])]
            await self._helper([*argv, "--height", str(FRAME[1])], folder, deadline, f"frame-{i}.png")
            if scene.caption:
                argv = [*base, "caption", "--text-file", f"cap-{i}.txt", "--out", f"cap-{i}.png"]
                await self._helper(argv, folder, deadline, f"cap-{i}.png")
        timeline = plan_timeline([s.length for s in scenes], plan.crossfade)
        for i, scene in enumerate(scenes):
            self._say(ctx, f"scene {i + 1}/{count}")
            argv = scene_argv(self.ffmpeg, i, timeline.clip_lengths[i], scene.motion, bool(scene.caption))
            await self._ffmpeg(argv, folder, deadline)
            if not _is_regular(folder / f"clip-{i}.mp4"):
                return _failed("ffmpeg wrote no video")
        self._say(ctx, "joining…")
        await self._ffmpeg(final_argv(self.ffmpeg, timeline, narrations, music_name, OUTPUT_NAME), folder, deadline)
        try:
            data, file_size = await asyncio.to_thread(_read_header, folder / OUTPUT_NAME, MAX_VIDEO_BYTES)
        except TooBig:
            return _failed("ffmpeg wrote a video over 200 MB")
        except (FileNotFoundError, NotADirectoryError):
            return _failed("ffmpeg wrote no video")
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):  # a symlink where the video should be
                return _failed("ffmpeg wrote no video")
            return _failed(f"Cannot read the video: {exc.strerror or exc}")
        info = mp4_info(data, file_size)
        if info is None:
            if file_size > len(data) and data[4:8] == b"ftyp":  # an MP4 whose moov is not in the part read
                return _failed("ffmpeg wrote an MP4 that is not faststart")
            return _failed("ffmpeg wrote something that is not an MP4 file")
        if (info.width, info.height) != FRAME:
            return _failed(f"ffmpeg wrote {info.width}\N{MULTIPLICATION SIGN}{info.height}, expected {FRAME[0]}\N{MULTIPLICATION SIGN}{FRAME[1]}")
        if not info.faststart:
            return _failed("ffmpeg wrote an MP4 that is not faststart")
        for seconds in (info.video_seconds, info.seconds):  # the video track, then the whole file
            if abs(seconds - timeline.total) > 0.2:
                return _failed(f"ffmpeg wrote {seconds:.1f} s of video, expected {timeline.total:.1f} s")
        try:
            refused = await finish_placement(asyncio.to_thread(place_file, path, folder / OUTPUT_NAME, ctx))
        except OSError as exc:
            return ToolResult(f"Cannot write {raw}: {exc.strerror or exc}", is_error=True, summary="write failed")
        if refused is not None:
            return refused
        word = "scene" if count == 1 else "scenes"
        size = f"{FRAME[0]}\N{MULTIPLICATION SIGN}{FRAME[1]}"
        described = f"MP4, {size}, {info.seconds:.1f} s, {count} {word}, {size_text(file_size)}"
        return ToolResult(f"assembled {raw} ({described})", summary=described)

    @staticmethod
    def _stage(plan: _Plan, folder: Path) -> tuple[str | None, list[str], list[str | None]]:
        """Write every input into the private folder (mode 0600) from the bytes the checks read: the image of scene i
        as `in-<i>.<ext>`, its narration as `nar-<i>.wav`, its caption as `cap-<i>.txt`, the music as `music.<kind>`.
        Returns the music name, the image names and the narration names."""
        images: list[str] = []
        narrations: list[str | None] = []
        for i, scene in enumerate(plan.scenes):
            images.append(stage_one(scene.image, folder, f"in-{i}").name)
            if scene.audio is not None:
                _write_bytes(folder / f"nar-{i}.wav", scene.audio.data)
                narrations.append(f"nar-{i}.wav")
            else:
                narrations.append(None)
            if scene.caption:
                _write_text(folder / f"cap-{i}.txt", scene.caption)
        music_name = None
        if plan.music is not None:
            music_name = f"music.{plan.music.kind}"
            _write_bytes(folder / music_name, plan.music.data)
        return music_name, images, narrations

    async def _helper(self, argv: list[str], folder: Path, deadline: float, output: str) -> None:
        code, stderr = await self._step(argv, folder, deadline, "the media helper")
        if code == 3:
            raise _Stop(_failed("an image could not be decoded"))
        if code == 5:
            raise _Stop(_failed("image too large for a video (limit 50 megapixels)"))
        if code != 0:
            raise _Stop(_failed(f"media helper failed (exit {code}): {_tail(stderr)}"))
        if not _is_regular(folder / output):
            raise _Stop(_failed("media helper wrote no image"))

    async def _ffmpeg(self, argv: list[str], folder: Path, deadline: float) -> None:
        code, stderr = await self._step(argv, folder, deadline, "ffmpeg")
        if code != 0:
            raise _Stop(_failed(f"ffmpeg failed (exit {code}): {_tail(stderr)}"))
