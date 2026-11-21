"""A fake mflux for the image tests: tiny PNGs and an executable script that behaves like the real command."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def png(width: int, height: int) -> bytes:
    """A valid PNG of the given size (a flat grey), built without Pillow."""

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


SCRIPT = r'''#!{python} -I
import json, os, struct, subprocess, sys, time, zlib

args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1]


def png(w, h):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    row = b"\x00" + b"\x80" * w
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))


mode = os.environ.get("FAKE_MFLUX_MODE", "ok")
dump = os.environ.get("FAKE_MFLUX_DUMP")
if dump:
    with open(dump, "w") as handle:
        json.dump({{"argv": args, "env": dict(os.environ), "cwd": os.getcwd()}}, handle)
out = opt("--output")
w, h = int(opt("--width")), int(opt("--height"))
if mode == "ok":
    open(out, "wb").write(png(w, h))
elif mode == "jpeg":
    open(out, "wb").write(b"\xff\xd8\xff\xe0" + b"\0" * 100)
elif mode == "truncated":
    open(out, "wb").write(png(w, h)[:-12])
elif mode == "wrongsize":
    open(out, "wb").write(png(w // 2, h // 2))
elif mode == "text":
    open(out, "wb").write(b"hello, not an image at all")
elif mode == "nothing":
    pass
elif mode == "fail":
    for n in range(int(os.environ.get("FAKE_MFLUX_LINES", "3"))):
        print("error line %d \x1b[2J" % n, file=sys.stderr)
    sys.exit(1)
elif mode == "token":
    print("boom ghp_" + "a" * 40, file=sys.stderr)
    sys.exit(1)
elif mode == "tqdm":
    sys.stderr.write("  0%|          | 0/9 [00:00<?, ?it/s]\r 11%|#1        | 1/9 [00:09<01:12, 9.00s/it]\r")
    sys.stderr.write("\nValueError: real reason\n")
    sys.exit(1)
elif mode == "sleep":
    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["FAKE_MFLUX_PIDS"], "w") as handle:
        handle.write("%d %d" % (os.getpid(), child.pid))
    open(out, "wb").write(b"partial")
    time.sleep(60)
'''


def fake_mflux(folder: Path) -> Path:
    """Write the fake command into `folder`; its behavior comes from `FAKE_MFLUX_MODE` (and friends) in the environment."""
    path = folder / "mflux-generate-z-image-turbo"
    path.write_text(SCRIPT.format(python=sys.executable))
    path.chmod(0o755)
    return path
