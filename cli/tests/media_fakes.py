"""A fake speech helper for the `speak` tests: an executable-free script that behaves like `speech_helper.py`."""

from __future__ import annotations

import struct
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
