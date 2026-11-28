"""SessionStart hooks in the daemon: once per session the daemon loads, for every client of that session."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from core.engine.provider.base import Done, TextDelta
from core.testing.helpers import wait_until

REPLY = [TextDelta("ok"), Done(1, 1)]


def _global_hook(command: str) -> Callable[[Path], None]:
    def setup(ws: Path) -> None:
        folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "settings.json").write_text(json.dumps({"hooks": {"SessionStart": [{"command": command, "timeout": 60}]}}))

    return setup


def _logging_hook(log: Path) -> str:
    code = f"import sys, json; d = json.load(sys.stdin); open({str(log)!r}, 'a').write(d['source'] + '\\n'); print(json.dumps({{'context': 'CTX'}}))"
    return f"{sys.executable} -c \"{code}\""


async def _open(served, client, source: str | None, resume: str | None = None) -> dict:
    got = await client.post("/v1/sessions", json={"resume": resume, "source": source})
    assert got.status_code == 200, got.text
    return got.json()


async def test_the_hook_runs_once_per_loaded_session_with_its_source_and_not_on_a_second_attach(serve, tmp_path) -> None:
    log = tmp_path / "log"
    served = await serve(setup=_global_hook(_logging_hook(log)))
    a, b = served.client("A"), served.client("B")
    first = await _open(served, a, "startup")
    await _open(served, b, "startup", resume=first["session_id"])  # a live session: nothing runs
    for source in ("new", "switch", "resume"):
        await _open(served, a, source)
    await wait_until(lambda: len(log.read_text().split()) >= 4 if log.exists() else False)
    await asyncio.sleep(0.3)
    assert log.read_text().split() == ["startup", "new", "switch", "resume"]


async def test_the_context_reaches_the_next_request_only_and_the_notice_every_client(serve, tmp_path) -> None:
    served = await serve([REPLY, REPLY], setup=_global_hook(_logging_hook(tmp_path / "log")))
    a, b = served.client("A"), served.client("B")
    session_id = (await _open(served, a, "startup"))["session_id"]
    sa, sb = served.stream(a, session_id, after=0), served.stream(b, session_id, after=0)
    for stream in (sa, sb):
        await stream.until(lambda s: any("hook SessionStart" in m.get("text", "") for m in s.of("notice")))
    host = served.daemon.host(session_id)
    await wait_until(lambda: not host.switching)
    for _ in range(2):
        await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
        await wait_until(lambda: not host.turn_active)
    first, second = (" ".join(str(m.content) for m in req[0]) for req in served.provider.requests)
    assert "CTX" in first and "CTX" not in second


def _slow(pid_file: Path) -> Callable[[Path], None]:
    return _global_hook(f"echo $$ > {pid_file}; exec sleep 30")


async def test_a_message_gets_the_busy_line_and_esc_from_another_client_kills_the_hook(serve, tmp_path) -> None:
    pid_file = tmp_path / "pid"
    served = await serve(setup=_slow(pid_file))
    a, b = served.client("A"), served.client("B")
    session_id = (await _open(served, a, "startup"))["session_id"]
    stream = served.stream(a, session_id, after=0)
    await wait_until(pid_file.exists)
    refused = await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "hi"})
    assert refused.json() == {"status": "refused", "reason": "busy — a session is loading"}
    await b.post(f"/v1/sessions/{session_id}/interrupt")
    await stream.until(lambda s: any(m.get("text") == "hook SessionStart interrupted" for m in s.of("notice")))
    pid = int(pid_file.read_text())

    def gone() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

    await wait_until(gone)


async def test_a_slow_hook_blocks_only_its_own_session(serve, tmp_path) -> None:
    pid_file = tmp_path / "pid"
    served = await serve([REPLY], setup=_slow(pid_file))
    a = served.client("A")
    slow_id = (await _open(served, a, "startup"))["session_id"]
    await wait_until(pid_file.exists)
    other = (await _open(served, a, None))["session_id"]  # no source: no hook
    stream = served.stream(a, other, after=0)
    await a.post(f"/v1/sessions/{other}/messages", json={"text": "hi"})
    await stream.until(lambda s: "turn_outcome" in s.types())
    started = time.monotonic()
    assert (await a.get(f"/v1/sessions/{slow_id}/info/usage")).status_code == 200
    assert time.monotonic() - started < 0.2
    await a.post(f"/v1/sessions/{slow_id}/interrupt")
