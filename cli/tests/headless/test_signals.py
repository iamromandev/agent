"""SIGINT and SIGTERM during a headless run: exit 130 / 143 and no child of the run is left (NFR1, AC10, AC11).

Each test runs `python -m src.main -p ...` against a tiny fake Ollama over HTTP whose first reply calls a tool that
starts a sleeping child; the child writes its PID, the test signals the agent and then checks that PID is gone."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

CLI = Path(__file__).parent.parent.parent


def _handler(chat_reply: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None: ...

        def _send(self, body: dict) -> None:
            data = (json.dumps(body) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send({"models": [{"name": "m"}]})

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/api/show":
                self._send({"model_info": {"general.architecture": "x", "x.context_length": 8192}})
            else:
                self._send(chat_reply)

    return Handler


@pytest.fixture
def ollama() -> Iterator[ThreadingHTTPServer]:
    reply = {
        "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "SLEEP"}}}]},
        "done": True,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(reply))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(path: Path, timeout: float = 20) -> int:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


def _launch(tmp_path: Path, server: ThreadingHTTPServer, *args: str, command: str = "", config: dict | None = None) -> subprocess.Popen[bytes]:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = tmp_path / "cfg"
    if config is not None:
        (cfg / "agent").mkdir(parents=True, exist_ok=True)
        for name, data in config.items():
            (cfg / "agent" / name).write_text(json.dumps(data))
    env = {
        **os.environ,
        "AGENT_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
        "AGENT_MODEL": "m",
        "XDG_CONFIG_HOME": str(cfg),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "AGENT_SESSIONS_DIR": str(tmp_path / "sessions"),
        "PYTHONPATH": str(CLI),
    }
    env.pop("AGENT_SEARCH_URL", None)
    if command:
        # Safe after `serve_forever` started: ThreadingHTTPServer builds a handler per request from this attribute
        # (`finish_request`), and nothing talks to it until the Popen below starts the agent.
        server.RequestHandlerClass = _handler(  # type: ignore[assignment]
            {
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}]},
                "done": True,
            }
        )
    return subprocess.Popen(
        [sys.executable, "-c", "from src.main import main; main()", "-p", "go", *args], cwd=ws, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _gone(pid: int, timeout: float = 5) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(("sig", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_a_signal_during_bash_exits_130_or_143_and_leaves_no_sleeping_child(
    tmp_path: Path, ollama: ThreadingHTTPServer, sig: signal.Signals, code: int
) -> None:
    pidfile = tmp_path / "pid"
    command = f"echo $$ > {pidfile}; exec sleep 30"
    proc = _launch(tmp_path, ollama, "--allow", f"bash({command})", command=command)
    try:
        pid = _wait_for(pidfile)
        assert _alive(pid)

        proc.send_signal(sig)
        out, err = proc.communicate(timeout=20)

        assert proc.returncode == code, err.decode()
        assert out == b""
        assert _gone(pid)
        assert b"Traceback" not in err
        files = list((tmp_path / "sessions").rglob("*.jsonl"))
        assert files and all(json.loads(line) for f in files for line in f.read_text().splitlines())  # flushed, whole lines
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_signal_during_a_hook_leaves_no_sleeping_child(tmp_path: Path, ollama: ThreadingHTTPServer) -> None:
    pidfile = tmp_path / "hookpid"
    hook = {"hooks": {"UserPromptSubmit": [{"command": f"echo $$ > {pidfile}; exec sleep 30"}]}}
    proc = _launch(tmp_path, ollama, config={"settings.json": hook})
    try:
        pid = _wait_for(pidfile)

        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=20)

        assert proc.returncode == 143
        assert _gone(pid)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_signal_while_an_mcp_server_starts_leaves_no_server_process(tmp_path: Path, ollama: ThreadingHTTPServer) -> None:
    pidfile = tmp_path / "mcppid"
    mcp = {"servers": {"slow": {"command": "sh", "args": ["-c", f"echo $$ > {pidfile}; exec sleep 30"]}}}
    proc = _launch(tmp_path, ollama, config={"mcp.json": mcp})
    try:
        pid = _wait_for(pidfile)

        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=20)

        assert proc.returncode == 130
        assert _gone(pid)
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.parametrize(("sig", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
@pytest.mark.parametrize("mode", ["json", "stream-json"])
def test_a_signal_in_json_mode_ends_with_one_interrupted_result(
    tmp_path: Path, ollama: ThreadingHTTPServer, sig: signal.Signals, code: int, mode: str
) -> None:
    pidfile = tmp_path / "pid"
    command = f"echo $$ > {pidfile}; exec sleep 30"
    proc = _launch(tmp_path, ollama, "--allow", f"bash({command})", "--output", mode, command=command)
    try:
        pid = _wait_for(pidfile)

        proc.send_signal(sig)
        out, err = proc.communicate(timeout=20)

        assert proc.returncode == code, err.decode()
        lines = [json.loads(line) for line in out.decode().splitlines()]
        last = lines[-1]
        assert (last["type"], last["exit_code"], last["stop_reason"], last["is_error"]) == ("result", code, "interrupted", True)
        assert [line["type"] for line in lines].count("result") == 1
        assert [t["name"] for t in last["tool_calls"]] == ["bash"] and last["tool_calls"][0]["summary"] == "interrupted"
        assert _gone(pid)
        assert b"Traceback" not in err
    finally:
        if proc.poll() is None:
            proc.kill()
