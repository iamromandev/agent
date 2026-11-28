"""A fake language server for the LSP tests: a stdlib-only script that speaks enough JSON-RPC over stdio.

`FAKE_LSP_MODE`: `pull` (offers `diagnosticProvider`), `push` (publishes with the version it was sent), `silent` (push
capable, never publishes), `crash` (exits on `initialize`), `garbage` (answers `initialize`, then writes a bad frame),
`flood` (a header past the 16 MiB limit), `hostile` (push: stale version, other files and a file outside the
workspace first), `late` (pull, answers after `FAKE_LSP_DELAY` seconds, default 30), `initerror` (answers `initialize`
with a JSON-RPC error), `slowinit` (answers `initialize` after `FAKE_LSP_DELAY` seconds), `spray` (push; first 5,000
publications for other files), `nover` (push, publishes without a `version`), `eof` (closes stdout after `initialize`
but keeps running).
`FAKE_LSP_REPORT` is a JSON list of `{line, character, severity, message, code}` (0-based position, as on the wire) or
`{"repeat": N, "item": {...}}`; the default is one error and one warning. `FAKE_LSP_DUMP` is a file that gets one
`{pid, cwd, pgid, method, params}` JSON line per message received (and a `start` line)."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

SCRIPT = r'''#!{python} -I
import json, os, sys, time

mode = os.environ.get("FAKE_LSP_MODE", "pull")
dump = os.environ.get("FAKE_LSP_DUMP")
out = sys.stdout.buffer
inp = sys.stdin.buffer


def note(method, params=True):
    if dump:
        with open(dump, "a") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "cwd": os.getcwd(), "pgid": os.getpgid(0), "method": method, "params": params}) + "\n")


def send(obj):
    body = json.dumps(obj).encode()
    try:
        out.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        out.flush()
    except Exception:
        pass  # the eof mode closed stdout


def read():
    length = None
    while True:
        line = inp.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        name, _, value = line.partition(b":")
        if name.lower() == b"content-length":
            length = int(value)
    return json.loads(inp.read(length))


def report():
    raw = json.loads(os.environ.get("FAKE_LSP_REPORT") or "null")
    if raw is None:
        raw = [
            {"line": 0, "character": 4, "severity": 1, "message": "bad thing", "code": "E1"},
            {"line": 1, "character": 0, "severity": 2, "message": "meh"},
        ]
    if isinstance(raw, dict):
        raw = [raw["item"]] * raw["repeat"]
    items = []
    for d in raw:
        item = {
            "range": {"start": {"line": d["line"], "character": d["character"]}, "end": {"line": d["line"], "character": d["character"] + 1}},
            "severity": d["severity"], "message": d["message"],
        }
        if d.get("code") is not None:
            item["code"] = d["code"]
        items.append(item)
    return items


note("start")
pull = mode in ("pull", "late")
current = {}  # uri -> version
waiting = []  # pull requests held until workspace/configuration is answered
configured = False
while True:
    msg = read()
    if msg is None:
        break
    method = msg.get("method")
    if method is not None:
        note(method, "params" in msg)
    if method == "initialize":
        if mode == "crash":
            sys.exit(3)
        if mode == "initerror":
            send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": "boom"}})
            continue
        if mode == "slowinit":
            time.sleep(float(os.environ.get("FAKE_LSP_DELAY", "30")))
        caps = {"textDocumentSync": 1}
        if pull:
            caps["diagnosticProvider"] = {"interFileDependencies": False, "workspaceDiagnostics": False}
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": caps}})
        if mode == "garbage":
            out.write(b"Content-Length: 5\r\n\r\nnope!")
            out.flush()
        if mode == "flood":
            out.write(b"Content-Length: 99999999999\r\n\r\n")
            out.flush()
        if mode == "eof":
            os.close(1)  # the interpreter's stdout does not own fd 1, so close the descriptor itself
    elif method == "initialized":
        send({"jsonrpc": "2.0", "id": "cfg", "method": "workspace/configuration", "params": {"items": [{"section": "x"}]}})
        send({"jsonrpc": "2.0", "id": "prog", "method": "window/workDoneProgress/create", "params": {"token": "t"}})
    elif method in ("textDocument/didOpen", "textDocument/didChange"):
        params = msg["params"]
        doc = params["textDocument"]
        uri, version = doc["uri"], doc["version"]
        current[uri] = version
        if mode in ("push", "hostile", "spray", "nover"):
            if mode == "spray":
                for n in range(5000):
                    send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": "file:///spray/%d.py" % n, "version": 1, "diagnostics": report()}})
            if mode == "nover":
                send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": uri, "diagnostics": report()}})
                continue
            if mode == "hostile":
                send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": uri, "version": version - 1, "diagnostics": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "severity": 1, "message": "stale"}]}})
                for other in ("file:///etc/passwd", uri + ".other.py"):
                    send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": other, "version": version, "diagnostics": report()}})
            send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": uri, "version": version, "diagnostics": report()}})
    elif method == "textDocument/diagnostic":
        if mode == "late":
            time.sleep(float(os.environ.get("FAKE_LSP_DELAY", "30")))
        if configured:
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"kind": "full", "items": report()}})
        else:
            waiting.append(msg["id"])
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        break
    elif method is None and msg.get("id") == "cfg":
        configured = True
        for pending in waiting:
            send({"jsonrpc": "2.0", "id": pending, "result": {"kind": "full", "items": report()}})
        waiting.clear()
    elif method is not None and "id" in msg:
        send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}})
'''


def fake_lsp(folder: Path, name: str = "ty") -> Path:
    """An executable fake server called `name` in `folder` (put the folder on PATH, or pass the path as the command)."""
    path = folder / name
    path.write_text(SCRIPT.replace("{python}", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
