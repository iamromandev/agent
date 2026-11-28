"""A fake formatter for the format tests: a stdlib-only script that works on the file named by its last argument.

Modes: `upper` (rewrite the file in upper case), `same` (touch nothing), `fail` (garble the file, then exit 1 with
a message on stderr), `sleep` (run for a minute), `empty` (truncate the file), `dump` (change nothing; append
`{argv, cwd, pgid, env}` to the file `FAKE_FORMAT_DUMP` names). Every mode but `dump` also appends the same line
when `FAKE_FORMAT_DUMP` is set."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

SCRIPT = r'''#!{python} -I
import json, os, sys, time

mode = "{mode}"
path = sys.argv[-1]
dump = os.environ.get("FAKE_FORMAT_DUMP")
if dump:
    with open(dump, "a") as handle:
        handle.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "pgid": os.getpgid(0), "env": sorted(os.environ)}) + "\n")
if mode == "upper":
    with open(path) as handle:
        text = handle.read()
    with open(path, "w") as handle:
        handle.write(text.upper())
elif mode == "fail":
    with open(path, "w") as handle:
        handle.write("garbled")
    sys.stderr.write("boom: cannot format\n")
    sys.exit(1)
elif mode == "sleep":
    time.sleep(60)
elif mode == "empty":
    open(path, "w").close()
'''


def fake_formatter(folder: Path, name: str, mode: str = "upper") -> Path:
    """An executable fake formatter called `name` in `folder`."""
    path = folder / name
    path.write_text(SCRIPT.replace("{python}", sys.executable).replace("{mode}", mode))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
