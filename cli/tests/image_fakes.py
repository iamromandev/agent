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
import hashlib, json, os, struct, subprocess, sys, time, zlib

args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1]


def png(w, h):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    color = os.environ.get("FAKE_MFLUX_COLOR")
    if color:
        rgb = bytes(int(v) for v in color.split(","))
        row = b"\x00" + rgb * w
        header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    else:
        row = b"\x00" + b"\x80" * w
        header = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))


mode = os.environ.get("FAKE_MFLUX_MODE", "ok")
dump = os.environ.get("FAKE_MFLUX_DUMP")
inputs = []
if "--image-paths" in args:
    for value in args[args.index("--image-paths") + 1:]:
        if value.startswith("--"):
            break
        inputs.append(value)
if dump:
    shas = {{}}
    for value in inputs:
        try:
            shas[value] = hashlib.sha256(open(value, "rb").read()).hexdigest()
        except OSError:
            shas[value] = None
    with open(dump, "w") as handle:
        json.dump({{"argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "inputs": inputs, "shas": shas}}, handle)
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
    sys.stderr.write("  0%|          | 0/9 [00:00<?, ?it/s]\r 11%|#1        | 1/9 [00:09<01:12, 9.00s/it]")
    sys.stderr.write("\nValueError: real reason\n")
    sys.exit(1)
elif mode in ("progress", "download", "badlines", "flood", "progressfail", "hugedigits"):
    total = int(opt("--steps"))
    delay = float(os.environ.get("FAKE_MFLUX_DELAY", "0"))

    def say(text):
        sys.stderr.write(text)
        sys.stderr.flush()

    def bar(n, left):
        pct = 100 * n // total
        say("\r%3d%%|%s| %d/%d [00:%02d<%s, 1.00s/it]" % (pct, "#" * (n * 10 // total), n, total, n, left))

    if mode == "download":
        for pct in (10, 42, 100):
            say("\rmodel.safetensors: %3d%%|####      | 2.3G/5.5G [00:30<00:41, 77MB/s]" % pct)
            time.sleep(delay)
        say("\n")
    if mode == "hugedigits":
        say("\r 1%|x| " + "9" * 5000 + "/9 [00:26<03:34, 1s/it]\n")
    if mode == "badlines":
        say("\xff\xfe garbage \x00\x01\n")
        say("Loading weights\n")
        say(" 50%|#####     | 5/")
        say("\n")
        bar(1, "00:09")
        say("\r100%|##########| 3/3 [00:03<00:00, 1.00s/it]")
        say("\rsome prefix 11%|#1 | 1/9 [00:01<00:08, 1.00s/it]")
    if mode == "flood":
        for _ in range(10000):
            say("\r  0%|          | 0/9 [00:00<?, ?it/s]")
    time.sleep(delay)
    for n in range(0, total + 1):
        bar(n, "?" if n == 0 else "00:%02d" % (total - n))
        time.sleep(delay)
    if mode == "badlines":
        bar(2, "00:07")
    say("\n")
    bar(total, "00:00")
    say("\nINFO: Image saved successfully at: %s\nPeak MLX memory: 4.54 GB\n" % out)
    if mode == "progressfail":
        say("ValueError: real reason\n")
        sys.exit(1)
    open(out, "wb").write(png(w, h))
elif mode in ("sleep", "stubborn"):
    if mode == "stubborn":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen(["sleep", "60"])
    with open(os.environ["FAKE_MFLUX_PIDS"], "w") as handle:
        handle.write("%d %d" % (os.getpid(), child.pid))
    open(out, "wb").write(b"partial")
    time.sleep(60)
'''


def fake_mflux(folder: Path, name: str = "mflux-generate-z-image-turbo") -> Path:
    """Write the fake command `name` into `folder`; its behavior comes from `FAKE_MFLUX_MODE` (and friends) in the environment."""
    path = folder / name
    path.write_text(SCRIPT.format(python=sys.executable))
    path.chmod(0o755)
    return path


COMPOSITE_SCRIPT = r'''import json, os, struct, sys, time, zlib

step, args = sys.argv[1], sys.argv[2:]


def opt(name):
    return args[args.index(name) + 1]


def png(w, h):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    row = b"\x00" + b"\x80" * w
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))


dump = os.environ.get("FAKE_COMPOSITE_DUMP")
if dump:
    with open(dump, "a") as handle:
        handle.write(json.dumps({"step": step, "argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "exe": sys.executable, "flags": sys.flags.isolated}) + "\n")
mode = os.environ.get("FAKE_COMPOSITE_MODE", "ok")
on = os.environ.get("FAKE_COMPOSITE_STEP", "blend")
if on in (step, "both") and mode != "ok":
    if mode == "fail":
        print("boom", file=sys.stderr)
        sys.exit(1)
    if mode == "nothing":
        sys.exit(3)
    if mode == "orientation":
        print("source is 20x10, expected 10x20", file=sys.stderr)
        sys.exit(4)
    if mode == "toolarge":
        print("image too large to composite (limit 50 megapixels)", file=sys.stderr)
        sys.exit(5)
    if mode == "sleep":
        with open(os.environ["FAKE_COMPOSITE_PIDS"], "w") as handle:
            handle.write(str(os.getpid()))
        time.sleep(60)
if step == "prepare":
    plant = os.environ.get("FAKE_COMPOSITE_PLANT")
    if plant:
        os.symlink(plant, os.path.join(os.getcwd(), "out.png"))
    open(opt("--out-mask"), "wb").write(png(4, 4))
else:
    if mode == "nofile":
        sys.exit(0)
    data = open(opt("--source"), "rb").read()
    w, h = struct.unpack(">II", data[16:24])
    if mode == "wrongsize":
        w, h = w + 1, h
    elif mode == "text":
        open(opt("--out"), "wb").write(b"not a png at all")
        sys.exit(0)
    open(opt("--out"), "wb").write(png(w, h))
'''


def fake_composite(folder: Path) -> Path:
    """A fake compositing helper script (run as `python -I <path> prepare|blend ...`). `FAKE_COMPOSITE_MODE` is ok,
    fail, nothing (exit 3), orientation (exit 4), toolarge (exit 5), sleep (`FAKE_COMPOSITE_PIDS`), nofile, wrongsize or text, applied to
    the step named by `FAKE_COMPOSITE_STEP` (blend by default; prepare or both); `FAKE_COMPOSITE_DUMP` appends one JSON
    line per run. Its blend output has the source's size (a PNG source)."""
    path = folder / "fake_composite.py"
    path.write_text(COMPOSITE_SCRIPT)
    return path


HF_SCRIPT = r'''#!{python} -I
import json, os, sys, time

dump = os.environ.get("FAKE_HF_DUMP")
if dump:
    with open(dump, "w") as handle:
        json.dump({{"argv": sys.argv[1:], "env": dict(os.environ), "pid": os.getpid()}}, handle)
time.sleep({sleep!r})
sys.stdout.write({stdout!r})
sys.stderr.write("Warning: You are sending unauthenticated requests to the HF Hub.\n")
sys.exit({code!r})
'''


def fake_hf(folder: Path, stdout: str, code: int = 0, sleep: float = 0.0) -> Path:
    """Write a fake `hf` into `folder` that prints `stdout`; argv and environment go to `FAKE_HF_DUMP` when set."""
    path = folder / "hf"
    path.write_text(HF_SCRIPT.format(python=sys.executable, stdout=stdout, code=code, sleep=sleep))
    path.chmod(0o755)
    return path


def dry_run(rows: list[tuple[str, str]], will: str = "9 files (out of 25) totalling 7.8G") -> str:
    """What `hf download <repo> --dry-run` prints: a summary line, then a tab-separated table."""
    return f"[dry-run] Will download {will}.\nfile\tsize\n" + "".join(f"{name}\t{size}\n" for name, size in rows)


COMPARE_SCRIPT = r'''import json, os, sys

dump = os.environ.get("FAKE_COMPARE_DUMP")
if dump:
    with open(dump, "a") as handle:
        handle.write(json.dumps(sys.argv[1:]) + "\n")
code = int(os.environ.get("FAKE_COMPARE_EXIT", "0"))
only = os.environ.get("FAKE_COMPARE_ONLY")
if only and sys.argv[1:2] != [only]:
    code = 0
if code:
    print("pixels differ" if code == 1 else "compare broke", file=sys.stderr)
sys.exit(code)
'''


def fake_compare(folder: Path) -> Path:
    """A fake `pixel_compare.py` (run as `python -I <path> <mode> ...`): exits with `FAKE_COMPARE_EXIT` (default 0), only
    for the mode `FAKE_COMPARE_ONLY` when set; each argv is appended as a JSON line to `FAKE_COMPARE_DUMP` when set."""
    path = folder / "fake_compare.py"
    path.write_text(COMPARE_SCRIPT)
    return path
