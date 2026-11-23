"""A fake speech helper for the `speak` tests: an executable-free script that behaves like `speech_helper.py`."""

from __future__ import annotations

import inspect
import struct
import sys
from pathlib import Path


def wav_bytes(seconds: float = 1.0, rate: int = 24000) -> bytes:
    """A valid mono PCM16 WAV of silence."""
    data = b"\x00\x00" * int(seconds * rate)
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


SCRIPT = r'''import json, os, struct, subprocess, sys, time

args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1]


def wav(seconds, rate=24000):
    data = b"\x00\x00" * int(seconds * rate)
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


mode = os.environ.get("FAKE_SPEECH_MODE", "ok")
text = open(opt("--text-file"), encoding="utf-8").read()
dump = os.environ.get("FAKE_SPEECH_DUMP")
if dump:
    with open(dump, "w") as handle:
        json.dump({
            "argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "text": text, "exe": sys.executable,
            "isolated": sys.flags.isolated, "text_mode": oct(os.stat(opt("--text-file")).st_mode & 0o777),
        }, handle)
out = opt("--out")
if mode == "ok":
    open(out, "wb").write(wav(float(os.environ.get("FAKE_SPEECH_SECONDS", "1.0"))))
    skipped = json.loads(os.environ.get("FAKE_SPEECH_SKIPPED", "[]"))
    json.dump(skipped, open(opt("--skipped"), "w"))
elif mode == "badskipped":
    open(out, "wb").write(wav(1.0))
    open(opt("--skipped"), "w").write("not json")
elif mode == "nothing":
    pass
elif mode == "notwav":
    open(out, "wb").write(b"ID3\x04" + b"\0" * 100)
elif mode == "silent":
    open(out, "wb").write(wav(0))
elif mode == "symlink":
    os.symlink(opt("--text-file"), out)
elif mode == "fail":
    for n in range(int(os.environ.get("FAKE_SPEECH_LINES", "3"))):
        print("error line %d \x1b[2J" % n, file=sys.stderr)
    sys.exit(1)
elif mode == "token":
    print("boom ghp_" + "a" * 40, file=sys.stderr)
    sys.exit(1)
elif mode == "model":
    print("could not load the model x: OSError", file=sys.stderr)
    sys.exit(3)
elif mode == "sleep":
    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["FAKE_SPEECH_PIDS"], "w") as handle:
        handle.write("%d %d" % (os.getpid(), child.pid))
    time.sleep(60)
'''


def fake_speech(folder: Path, name: str = "speech_helper.py") -> Path:
    """Write the fake helper into `folder`; its behavior comes from `FAKE_SPEECH_MODE` (and friends) in the environment."""
    path = folder / name
    path.write_text(SCRIPT)
    return path


FFMPEG_SCRIPT = r'''import json, os, struct, subprocess, sys, time

args = sys.argv[1:]
out = args[-1]
scene = out.startswith("clip-")
kind = "scene" if scene else "final"
mode = os.environ.get("FAKE_FFMPEG_MODE", "ok")
only = os.environ.get("FAKE_FFMPEG_ONLY", "")
if only and only != kind:
    mode = "ok"
dump = os.environ.get("FAKE_FFMPEG_DUMP")
if dump:
    inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    with open(dump, "a") as handle:
        handle.write(json.dumps({
            "kind": kind, "argv": args, "env": dict(os.environ), "cwd": os.getcwd(),
            "inputs": {name: os.path.isfile(name) for name in inputs},
        }) + "\n")
if mode == "ok" or mode in ("wrongsize", "wronglen", "shortvideo", "notmp4", "slowstart"):
    if scene:
        open(out, "wb").write(b"fake clip " + out.encode())
    elif mode == "notmp4":
        open(out, "wb").write(b"this is not an mp4 file at all" * 4)
    else:
        seconds = float(args[args.index("-t") + 1])
        width, height = (720, 1280) if mode == "wrongsize" else (1080, 1920)
        open(out, "wb").write(
            mp4_bytes(
                width, height, seconds + (5 if mode == "wronglen" else 0), faststart=mode != "slowstart",
                video_seconds=seconds - 1 if mode == "shortvideo" else None,
            )
        )
elif mode == "nothing":
    pass
elif mode == "fail":
    for n in range(3):
        print("error line %d \x1b[2J" % n, file=sys.stderr)
    sys.exit(1)
elif mode == "token":
    print("boom ghp_" + "a" * 40, file=sys.stderr)
    sys.exit(1)
elif mode == "sleep":
    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["FAKE_FFMPEG_PIDS"], "w") as handle:
        handle.write("%d %d" % (os.getpid(), child.pid))
    time.sleep(60)
'''

MEDIA_SCRIPT = r'''import json, os, struct, subprocess, sys, time, zlib

args = sys.argv[1:]
step = args[0]


def opt(name):
    return args[args.index(name) + 1]


def png(w, h, rgba=False):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    row = b"\x00" + (b"\x00\x00\x00\x00" if rgba else b"\x80\x80\x80") * w
    header = struct.pack(">IIBBBBB", w, h, 8, 6 if rgba else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * h))
            + chunk(b"IEND", b""))


mode = os.environ.get("FAKE_MEDIA_MODE", "ok")
only = os.environ.get("FAKE_MEDIA_ONLY", "")
if only and only != step:
    mode = "ok"
dump = os.environ.get("FAKE_MEDIA_DUMP")
text = open(opt("--text-file"), encoding="utf-8").read() if step == "caption" else None
if dump:
    with open(dump, "a") as handle:
        handle.write(json.dumps({
            "step": step, "argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "text": text,
            "exe": sys.executable, "isolated": sys.flags.isolated,
            "in_mode": oct(os.stat(opt("--in" if step == "frame" else "--text-file")).st_mode & 0o777),
        }) + "\n")
time.sleep(float(os.environ.get("FAKE_MEDIA_DELAY", "0")))
out = opt("--out")
if mode == "ok":
    if step == "frame":
        open(out, "wb").write(png(int(opt("--width")), int(opt("--height"))))
    else:
        open(out, "wb").write(png(1080, 1920, rgba=True))
elif mode == "nothing":
    pass
elif mode == "fail":
    print("helper broke \x1b[2J", file=sys.stderr)
    sys.exit(1)
elif mode == "undecodable":
    sys.exit(3)
elif mode == "toolarge":
    print("image too large for a video (limit 50 megapixels)", file=sys.stderr)
    sys.exit(5)
elif mode == "sleep":
    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["FAKE_MEDIA_PIDS"], "w") as handle:
        handle.write("%d %d" % (os.getpid(), child.pid))
    time.sleep(60)
'''


def fake_ffmpeg(folder: Path, name: str = "ffmpeg-fake") -> Path:
    """An executable fake ffmpeg in `folder`. A scene run (output `clip-<i>.mp4`) writes a small file; the final run
    writes a minimal MP4 of `-t` seconds at 1080x1920. `FAKE_FFMPEG_MODE` picks a failure (`wrongsize`, `wronglen`,
    `shortvideo` (the video track is 1 s shorter than the file), `notmp4`, `nothing`, `fail`, `sleep`, `token`),`FAKE_FFMPEG_ONLY` (`scene` or `final`) limits it to one kind of run
    and `FAKE_FFMPEG_DUMP` collects one JSON line per run."""
    path = folder / name
    path.write_text(f"#!{sys.executable}\n{inspect.getsource(mp4_bytes)}\n\n{FFMPEG_SCRIPT}")
    path.chmod(0o755)
    return path


def fake_media(folder: Path, name: str = "media_helper.py") -> Path:
    """A fake media helper: `frame` and `caption` write PNGs of the right size. `FAKE_MEDIA_MODE` picks `fail`,
    `undecodable` (exit 3), `toolarge` (exit 5), `nothing` or `sleep`; `FAKE_MEDIA_ONLY` limits it to one step and
    `FAKE_MEDIA_DUMP` collects one JSON line per run."""
    path = folder / name
    path.write_text(MEDIA_SCRIPT)
    return path


def mp4_bytes(
    width: int = 1080, height: int = 1920, seconds: float = 7.6, version: int = 0, faststart: bool = True,
    audio: bool = True, timescale: int = 1000, video_seconds: float | None = None, mdhd: bool = True,
    track_timescale: int = 15360,
) -> bytes:
    """A minimal MP4: ftyp, moov (mvhd, a video trak, optionally an audio trak with a 0x0 tkhd) and mdat. Each trak has
    `mdia` with `hdlr` and (unless `mdhd` is False) `mdhd`; the video track lasts `video_seconds` (default `seconds`)
    in its own `track_timescale`, and its `tkhd` duration is in the movie timescale. Self-contained (no module globals) because the fake ffmpeg embeds its source."""
    import struct

    def box(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", 8 + len(body)) + kind + body

    def tkhd(w: int, h: int, v: int, ticks: int) -> bytes:
        tail = b"\0" * 8 + b"\0" * 8 + b"\0" * 36
        ids = struct.pack(">QQIIQ", 0, 0, 1, 0, ticks) if v else struct.pack(">IIIII", 0, 0, 1, 0, ticks)
        return box(b"tkhd", bytes([v, 0, 0, 3]) + ids + tail + struct.pack(">II", w << 16, h << 16))

    def mdia(handler: bytes, length: float) -> bytes:
        parts = b""
        if mdhd:
            n = round(length * track_timescale)
            if version:
                body = bytes([1, 0, 0, 0]) + struct.pack(">QQIQ", 0, 0, track_timescale, n)
            else:
                body = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, track_timescale, n)
            parts += box(b"mdhd", body + b"\x55\xc4\0\0")
        parts += box(b"hdlr", b"\0" * 8 + handler + b"\0" * 12 + b"handler\0")
        return box(b"mdia", parts)

    ticks = round(seconds * timescale)
    shown = seconds if video_seconds is None else video_seconds
    if version:
        mvhd_body = bytes([1, 0, 0, 0]) + struct.pack(">QQIQ", 0, 0, timescale, ticks)
    else:
        mvhd_body = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, timescale, ticks)
    mvhd = box(b"mvhd", mvhd_body + b"\0" * 80)
    traks = box(b"trak", tkhd(0, 0, version, ticks) + mdia(b"soun", seconds)) if audio else b""
    traks += box(b"trak", tkhd(width, height, version, round(shown * timescale)) + mdia(b"vide", shown))
    moov = box(b"moov", mvhd + traks)
    ftyp = box(b"ftyp", b"isom" + b"\0\0\2\0" + b"isomiso2avc1mp41")
    mdat = box(b"mdat", b"\0" * 64)
    return ftyp + (moov + mdat if faststart else mdat + moov)
