"""Media tools' shared facts: the speech engine, the voices, and reading audio headers.

The engine (mlx-audio, misaki) is an optional extra and is never imported here: it runs in a helper process.
UI-free: no `textual`, no `httpx`.
"""

from __future__ import annotations

import importlib.util
import math
import os
import platform as platform_module
import stat
import struct
import sys
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.engine.tools.base import ToolContext, ToolResult

VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
DEFAULT_VOICE = "af_heart"
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 0.5, 2.0, 1.0
MAX_TEXT = 4000
MAX_AUDIO_BYTES = 50 * 1024 * 1024
TEXT_NAME, WAV_NAME, SKIPPED_NAME = "text.txt", "speech.wav", "skipped.json"
_MAX_CHUNKS = 64  # RIFF chunks looked at before giving up: bounds the walk over a hostile file

FRAME = (1080, 1920)
FPS = 30
ZOOM = 1.15
OVERSAMPLE = 2  # zoompan works on a frame this many times larger, so the slow zoom does not stair-step
MAX_SCENES, MAX_SCENE_S, MAX_TOTAL_S = 30, 60, 180
MIN_SCENE_S = 1.0
MAX_CAPTION, MAX_CAPTION_LINES, CAPTION_WIDTH = 200, 4, 32
MOTIONS = ("zoom-in", "zoom-out", "pan-left", "pan-right", "none")
DEFAULT_MOTION = "zoom-in"
MAX_CROSSFADE, DEFAULT_CROSSFADE = 1.0, 0.5
MUSIC_DB = -18
MUSIC_FADE_S = 2.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_INPUT_BYTES = 300 * 1024 * 1024  # every image, narration and the music together, as staged (once per scene)
MAX_PIXELS = 50_000_000
OUTPUT_NAME = "out.mp4"

_SPEC = importlib.util.find_spec("core.engine.speech_helper")  # found, never imported: it needs the media extra
SPEECH_HELPER: Path | None = Path(_SPEC.origin) if _SPEC is not None and _SPEC.origin else None


_VIDEO_SPEC = importlib.util.find_spec("core.engine.media_helper")  # found, never imported: it needs Pillow
MEDIA_HELPER: Path | None = Path(_VIDEO_SPEC.origin) if _VIDEO_SPEC is not None and _VIDEO_SPEC.origin else None


def find_speech(system: str = sys.platform, machine: str | None = None) -> Path | None:
    """The speech helper script when the engine is installed (Apple Silicon macOS, mlx-audio and misaki importable)."""
    machine = platform_module.machine() if machine is None else machine
    if system != "darwin" or machine != "arm64":
        return None
    try:
        if importlib.util.find_spec("mlx_audio") is None or importlib.util.find_spec("misaki") is None:
            return None
    except (ValueError, ImportError):  # a module with `__spec__ = None`, or a broken parent package
        return None
    helper = SPEECH_HELPER
    return helper if helper is not None and helper.is_file() else None


def find_ffmpeg(system: str = sys.platform, machine: str | None = None) -> Path | None:
    """The ffmpeg binary that `imageio-ffmpeg` (the `media` extra) ships, on Apple Silicon macOS: the first executable
    `binaries/ffmpeg-*` in its folder. The package is found, never imported."""
    machine = platform_module.machine() if machine is None else machine
    if system != "darwin" or machine != "arm64":
        return None
    try:
        spec = importlib.util.find_spec("imageio_ffmpeg")
    except (ValueError, ImportError):  # a module with `__spec__ = None`, or a broken parent package
        return None
    if spec is None or not spec.origin:
        return None
    try:
        for candidate in sorted((Path(spec.origin).parent / "binaries").glob("ffmpeg-*")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    except OSError:
        return None
    return None


def sniff_audio(head: bytes) -> str | None:
    """`wav`, `mp3` or `m4a` from the first bytes of a file, or None. The name is never used."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and (head[1] >> 3) & 3 != 1 and (head[1] >> 1) & 3 != 0:
        return "mp3"  # a frame sync with neither the reserved MPEG version (01) nor the reserved layer (00)
    if head[4:8] == b"ftyp" and head[8:12] in (b"M4A ", b"mp42", b"isom"):
        return "m4a"
    return None


@dataclass(frozen=True)
class InputAudio:
    label: str  # escaped for display
    kind: str  # `wav`, `mp3` or `m4a`, from the bytes
    data: bytes = field(repr=False)
    identity: tuple[int, int]  # (st_dev, st_ino)


_KIND_WORDS = {"wav": "WAV", "mp3": "MP3", "m4a": "M4A"}


def resolve_audio(ref: str, ctx: ToolContext, kinds: Sequence[str]) -> InputAudio | ToolResult:
    """Check one workspace audio file and read its bytes once: inside the workspace on the resolved path and on the
    opened file's real path, not protected (refused without a dialog: its bytes go to another process), opened with
    `O_NOFOLLOW`, a regular file, at most `MAX_AUDIO_BYTES`, and one of `kinds` by its first bytes (never its name).
    `clipboard:N` is not accepted. An error is a `ToolResult`."""
    from core.engine.guardrails.paths import fd_path  # not at module level: these import the tools package
    from core.engine.guardrails.setup import guard_of
    from core.engine.text import escape_display
    from core.engine.tools.base import ToolResult
    from core.engine.tools.fs import _refuse_outside
    from core.engine.tools.paths import InvalidPath, inside_workspace, invalid_path_result, resolve_path
    from core.engine.tools.protected import refuse_protected

    shown = escape_display(ref)

    def error(message: str, summary: str) -> ToolResult:
        return ToolResult(message, is_error=True, summary=summary)

    if ref.startswith("clipboard:"):
        return error(f"{shown}: audio cannot come from the clipboard; use a workspace file", "not a file")
    try:
        path = resolve_path(ctx, ref)
    except InvalidPath as exc:
        return invalid_path_result(exc)
    if not inside_workspace(path, ctx):
        return _refuse_outside()
    guard = guard_of(ctx)
    if guard.paths.is_protected(path):
        return refuse_protected()
    if path.is_dir():
        return error(f"{shown} is a directory", "directory")
    if not path.exists():
        return error(f"No such file: {shown}", "not found")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)  # NONBLOCK: a FIFO must not hang the open
    except OSError as exc:
        return error(f"Cannot read {shown}: {escape_display(exc.strerror or str(exc))}", "unreadable")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return error(f"{shown} is not a regular file", "not a regular file")
        where = fd_path(fd)  # what was opened, not what was checked: a folder above may have been swapped
        if where is None or not inside_workspace(Path(where), ctx):
            return _refuse_outside()
        if guard.paths.is_protected_fd(fd):
            return refuse_protected()
        kind = sniff_audio(os.pread(fd, 12, 0))
        if kind is None or kind not in kinds:
            wanted = ", ".join(_KIND_WORDS[k] for k in kinds)
            wanted = wanted.rsplit(", ", 1)[0] + " or " + wanted.rsplit(", ", 1)[1] if len(kinds) > 1 else wanted
            return error(f"{shown} is not a {wanted} file", "not audio")
        if info.st_size > MAX_AUDIO_BYTES:
            return error(f"{shown}: audio too large (limit {MAX_AUDIO_BYTES // (1024 * 1024)} MB)", "audio too large")
        chunks: list[bytes] = []
        have = 0
        while have < info.st_size:
            chunk = os.pread(fd, min(1024 * 1024, info.st_size - have), have)
            if not chunk:
                break
            chunks.append(chunk)
            have += len(chunk)
        return InputAudio(shown, kind, b"".join(chunks), (info.st_dev, info.st_ino))
    except OSError as exc:
        return error(f"Cannot read {shown}: {escape_display(exc.strerror or str(exc))}", "unreadable")
    finally:
        os.close(fd)


@dataclass(frozen=True)
class WavInfo:
    rate: int
    channels: int
    bits: int
    seconds: float


_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


def wav_info(data: bytes) -> WavInfo | None:
    """Sample rate, channels, bits and length of a PCM (16/24/32 bit) or float32 WAV, or None when it is malformed,
    of another format, or holds no audio. The length comes from the data actually present."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    fmt: tuple[int, int, int, int, int] | None = None  # code, channels, rate, byte rate, bits
    for _ in range(_MAX_CHUNKS):
        if pos + 8 > len(data):
            return None
        name = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if name == b"fmt ":
            if size < 16 or body + 16 > len(data):
                return None
            code, channels, rate, byte_rate, _align, bits = struct.unpack_from("<HHIIHH", data, body)
            if code == _EXTENSIBLE:
                if size < 40 or body + 26 > len(data):
                    return None
                code = struct.unpack_from("<H", data, body + 24)[0]
            fmt = (code, channels, rate, byte_rate, bits)
        elif name == b"data":
            if fmt is None:
                return None
            code, channels, rate, byte_rate, bits = fmt
            ok = (code == _PCM and bits in (16, 24, 32)) or (code == _FLOAT and bits == 32)
            if not ok or channels < 1 or rate < 1:
                return None
            width = channels * bits // 8
            present = min(size, len(data) - body)
            seconds = (present - present % width) / (rate * width)
            return WavInfo(rate, channels, bits, seconds) if seconds > 0 else None
        pos = body + size + (size & 1)
    return None


@dataclass(frozen=True)
class Timeline:
    starts: tuple[float, ...]  # when each narration begins in the final video
    clip_lengths: tuple[float, ...]  # how long each scene clip is rendered (it overlaps its neighbours by the fade)
    offsets: tuple[float, ...]  # where each xfade begins: `offsets[i-1]` joins clip i to what came before
    total: float
    crossfade: float = 0.0


def plan_timeline(lengths: Sequence[float], crossfade: float) -> Timeline:
    """The scene timing for scene lengths `a_i` and a crossfade `x`: narration i starts at `starts[i-1] + a[i-1] + x`,
    the fade into scene i begins at `starts[i] - x`, a clip is its scene plus a fade on each inner side, and the total
    is `sum(a) + (n - 1) * x`. Everything is counted in whole frames: each scene length is rounded up to the frame
    grid (up, so a narration is never cut and never overlaps the next scene; at least one frame) and the crossfade
    is rounded to the nearest frame (at least one frame when it is above 0), so the starts, offsets, clip lengths and
    total all sit on frames and the narration stays aligned with its picture."""
    n = len(lengths)
    scenes = [max(1, math.ceil(a * FPS - 1e-6)) for a in lengths]
    x = 0 if crossfade <= 0 else max(1, round(crossfade * FPS))
    starts = [0]
    for i in range(1, n):
        starts.append(starts[-1] + scenes[i - 1] + x)
    clips = [scenes[i] + (x if i > 0 else 0) + (x if i < n - 1 else 0) for i in range(n)]
    offsets = [starts[i] - x for i in range(1, n)]
    total = sum(scenes) + (n - 1) * x

    def secs(frames: int) -> float:
        return round(frames / FPS, 6)

    return Timeline(
        tuple(secs(s) for s in starts), tuple(secs(c) for c in clips), tuple(secs(o) for o in offsets), secs(total), secs(x)
    )


def format_text(text: str) -> str | None:
    """A caption wrapped for the frame: whitespace collapsed, lines of at most `CAPTION_WIDTH` characters (a longer
    word is broken), or None when it needs more than `MAX_CAPTION_LINES` lines. Empty text gives ''."""
    lines = textwrap.wrap(" ".join(text.split()), CAPTION_WIDTH, break_long_words=True, break_on_hyphens=False)
    return "\n".join(lines) if len(lines) <= MAX_CAPTION_LINES else None


_QUIET = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
_VIDEO_ENCODER = ["-c:v", "libx264", "-preset", "medium"]


def _read_input(name: str, extra: Sequence[str] = ()) -> list[str]:
    return ["-protocol_whitelist", "file", *extra, "-i", name]  # only the `file` protocol, whatever the name says


def scene_argv(ffmpeg: str | Path, i: int, length: float, motion: str, caption: bool) -> list[str]:
    """The ffmpeg run that renders scene `i` as `clip-<i>.mp4`: a slow pan or zoom over `frame-<i>.png`, with
    `cap-<i>.png` laid over it when `caption`. Built only from the validated numbers, the motion and fixed names."""
    frames = round(length * FPS)
    wide = (round(FRAME[0] * ZOOM * OVERSAMPLE), round(FRAME[1] * ZOOM * OVERSAMPLE))
    grow = f"{ZOOM - 1:g}"
    centre = "x='iw/2-(iw/zoom/2)'"
    if motion == "zoom-in":
        z, x = f"1+{grow}*on/{frames}", centre
    elif motion == "zoom-out":
        z, x = f"{ZOOM:g}-{grow}*on/{frames}", centre
    elif motion == "pan-left":
        z, x = f"{ZOOM:g}", f"x='(iw-iw/zoom)*(1-on/{frames})'"
    elif motion == "pan-right":
        z, x = f"{ZOOM:g}", f"x='(iw-iw/zoom)*on/{frames}'"
    else:
        z, x = "1", centre
    pan = f"zoompan=z='{z}':{x}:y='ih/2-(ih/zoom/2)':d=1:s={FRAME[0]}x{FRAME[1]}:fps={FPS},format=yuv420p"
    graph = f"[0:v]scale={wide[0]}:{wide[1]}:flags=lanczos,{pan}"
    graph += "[bg];[bg][1:v]overlay=0:0:format=auto,format=yuv420p[v]" if caption else "[v]"
    loop = ["-loop", "1", "-framerate", str(FPS), "-t", f"{length:.3f}"]
    argv = [str(ffmpeg), *_QUIET, *_read_input(f"frame-{i}.png", loop)]
    if caption:
        argv += _read_input(f"cap-{i}.png", loop)
    return [
        *argv, "-filter_complex", graph, "-map", "[v]", "-frames:v", str(frames), *_VIDEO_ENCODER,
        "-crf", "14", "-pix_fmt", "yuv420p", f"clip-{i}.mp4",
    ]  # fmt: skip


def final_argv(
    ffmpeg: str | Path, timeline: Timeline, narrations: Sequence[str | None], music: str | None, out: str
) -> list[str]:
    """The ffmpeg run that joins the scene clips (crossfaded, or cut when the fade is 0), mixes each scene's narration
    (`narrations[i]`, None for a silent scene) at its start and a looped music bed 18 dB under it, and writes `out`
    with the faststart layout and exactly `timeline.total` seconds."""
    n = len(timeline.clip_lengths)
    total = f"{timeline.total:.3f}"
    argv = [str(ffmpeg), *_QUIET]
    for i in range(n):
        argv += _read_input(f"clip-{i}.mp4")
    spoken = [(i, name) for i, name in enumerate(narrations) if name is not None]
    for _, name in spoken:
        argv += _read_input(name)
    if music is not None:
        argv += _read_input(music, ["-stream_loop", "-1"])
    parts: list[str] = []
    if n == 1:
        parts.append("[0:v]null[v]")
    elif timeline.crossfade > 0:
        fade = f"{timeline.crossfade:.3f}"
        previous = "[0:v]"
        for i in range(1, n):
            label = f"[x{i}]" if i < n - 1 else "[v]"
            parts.append(f"{previous}[{i}:v]xfade=transition=fade:duration={fade}:offset={timeline.offsets[i - 1]:.3f}{label}")
            previous = label
    else:
        parts.append("".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    if spoken:
        for slot, (i, _) in enumerate(spoken):
            delay = round(timeline.starts[i] * 1000)
            parts.append(f"[{n + slot}:a]adelay={delay}:all=1,aresample=48000[a{i}]")
        mix = "".join(f"[a{i}]" for i, _ in spoken) + f"amix=inputs={len(spoken)}:normalize=0"
        parts.append(f"{mix},atrim=0:{total}[aout]" if music is None else f"{mix}[voice]")
    if music is not None:
        music_at = n + len(spoken)
        fade_at = max(0.0, timeline.total - MUSIC_FADE_S)
        fade = min(MUSIC_FADE_S, timeline.total)
        bed = f"[{music_at}:a]aresample=48000,volume={MUSIC_DB}dB,afade=t=out:st={fade_at:.3f}:d={fade:.3f},atrim=0:{total}"
        if not spoken:
            parts.append(f"{bed}[aout]")
        else:
            parts += [f"{bed}[bed]", f"[voice][bed]amix=inputs=2:normalize=0,atrim=0:{total}[aout]"]
    audio = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"] if spoken or music is not None else ["-an"]
    return [
        *argv, "-filter_complex", ";".join(parts), "-map", "[v]", *audio, *_VIDEO_ENCODER, "-crf", "20", "-maxrate", "6M",
        "-bufsize", "12M", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", total, out,
    ]  # fmt: skip


@dataclass(frozen=True)
class Mp4Info:
    width: int
    height: int
    seconds: float
    faststart: bool  # moov comes before mdat, so playback can start before the whole file is read
    video_seconds: float  # the video track's own length (the file's `seconds` is the longest track)


_MAX_BOXES = 256  # boxes looked at per level: bounds the walk over a hostile file


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, body start, body end) of the boxes between `start` and `end`; stops at the first one that is malformed."""
    pos = start
    for _ in range(_MAX_BOXES):
        if pos + 8 > min(end, len(data)):
            return
        size, kind = struct.unpack_from(">I4s", data, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            return
        yield kind, pos + head, pos + size
        pos += size


def _mvhd(data: bytes, start: int, end: int) -> float | None:
    if end - start < 20:
        return None
    version = data[start]
    if version == 0 and end - start >= 20:
        timescale, ticks = struct.unpack_from(">II", data, start + 12)
    elif version == 1 and end - start >= 32:
        timescale = struct.unpack_from(">I", data, start + 20)[0]
        ticks = struct.unpack_from(">Q", data, start + 24)[0]
    else:
        return None
    return ticks / timescale if timescale else None


def _tkhd(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    if end <= start:
        return None
    version = data[start]
    at = start + (88 if version == 1 else 76)  # after the version and flags, the ids, and the layer and matrix
    if version not in (0, 1) or at + 8 > end:
        return None
    width, height = struct.unpack_from(">II", data, at)
    return width >> 16, height >> 16  # 16.16 fixed point


def _tkhd_ticks(data: bytes, start: int, end: int) -> int | None:
    """The duration of a `tkhd` (in the movie timescale)."""
    if end <= start:
        return None
    version = data[start]
    at = start + (28 if version == 1 else 20)
    if version not in (0, 1) or at + (8 if version == 1 else 4) > end:
        return None
    return int(struct.unpack_from(">Q" if version == 1 else ">I", data, at)[0])


def _handler(data: bytes, start: int, end: int) -> bytes | None:
    """The handler type of an `hdlr` (`vide`, `soun`, ...)."""
    return data[start + 8 : start + 12] if end - start >= 12 else None


_MAX_TRAKS = 8  # tracks looked at: real files have two, so a hostile one cannot make the walk long


def mp4_info(data: bytes, size: int | None = None) -> Mp4Info | None:
    """Size and length of an MP4 from its top-level boxes: `moov/mvhd` (the whole file), the first track with a
    picture size and the length of the video track (`trak` whose `mdia/hdlr` is `vide`, from `mdia/mdhd` in the
    track's own timescale, or from `tkhd` in the movie timescale when there is no `mdhd`). None when it is no MP4, is
    cut short or has no readable video length. `data` may be only the start of a file of `size` bytes: `moov` must be
    whole in it, `mdat` need not be. Walks a bounded number of boxes and reads nothing else."""
    try:
        total = len(data) if size is None else size
        moov: tuple[int, int] | None = None
        mdat_at: int | None = None
        for kind, body, stop in _boxes(data, 0, total):
            if kind == b"moov" and moov is None and stop <= len(data):
                moov = (body, stop)
            elif kind == b"mdat" and mdat_at is None:
                mdat_at = body
        if moov is None:
            return None
        seconds: float | None = None
        picture: tuple[int, int] | None = None
        video: float | None = None
        scale = None
        traks = 0
        for kind, body, stop in _boxes(data, *moov):
            if kind == b"mvhd" and seconds is None:
                seconds = _mvhd(data, body, stop)
                scale = struct.unpack_from(">I", data, body + (20 if data[body] == 1 else 12))[0] if seconds is not None else None
            elif kind == b"trak" and traks < _MAX_TRAKS:
                traks += 1
                found = ticks = media_seconds = handler = None
                has_mdhd = False
                for inner, inner_body, inner_stop in _boxes(data, body, stop):
                    if inner == b"tkhd":
                        found = _tkhd(data, inner_body, inner_stop)
                        ticks = _tkhd_ticks(data, inner_body, inner_stop)
                    elif inner == b"mdia":
                        for leaf, leaf_body, leaf_stop in _boxes(data, inner_body, inner_stop):
                            if leaf == b"mdhd" and not has_mdhd:
                                has_mdhd = True
                                media_seconds = _mvhd(data, leaf_body, leaf_stop)
                            elif leaf == b"hdlr" and handler is None:
                                handler = _handler(data, leaf_body, leaf_stop)
                if picture is None and found is not None and found[0] and found[1]:
                    picture = found
                if video is None and handler == b"vide":
                    if has_mdhd:
                        video = media_seconds
                    elif ticks is not None and scale:
                        video = ticks / scale
        if seconds is None or picture is None or video is None:
            return None
        return Mp4Info(picture[0], picture[1], seconds, mdat_at is not None and moov[0] < mdat_at, video)
    except struct.error:
        return None
