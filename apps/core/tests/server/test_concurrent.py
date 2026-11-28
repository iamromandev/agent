"""Sessions of one daemon run turns at the same time; `/undo` leaves what another session's overlapping turn changed."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import Gate
from core.testing.helpers import wait_until

REPLY = [TextDelta("ok"), Done(1, 1)]
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def allow_writes(_ws: Path) -> None:
    folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps({"permissions": {"allow": ["write"]}}))


def writes(call_prefix: str, **files: str) -> list[Any]:
    calls = [ToolCallEvent(ToolCall(f"{call_prefix}{i}", "write", {"path": name.replace("_dot_", "."), "content": body})) for i, (name, body) in enumerate(files.items())]
    return [*calls, Done(1, 1)]


async def open_sessions(served: Any, count: int) -> list[tuple[Any, str]]:
    out = []
    for name in "ABC"[:count]:
        client = served.client(name)
        out.append((client, (await served.open(client))["session_id"]))
    return out


async def idle(served: Any, session_id: str) -> None:
    host = served.daemon.host(session_id)
    await wait_until(lambda: not host.turn_active)


async def send(client: Any, session_id: str, text: str) -> dict[str, Any]:
    return (await client.post(f"/v1/sessions/{session_id}/messages", json={"text": text})).json()


async def undo_text(client: Any, session_id: str) -> str:
    body = (await client.post(f"/v1/sessions/{session_id}/undo")).json()
    return " ".join(str(e.get("text", "")) for e in body["events"])


async def test_three_sessions_run_turns_at_the_same_time_and_none_is_refused(serve) -> None:
    # Bug caught: a turn in one session still refused (or made to wait) because another session is running one.
    gate_a, gate_b = Gate(), Gate()
    served = await serve([[gate_a, *REPLY], [gate_b, *REPLY], REPLY])
    (a, one), (b, two), (c, three) = await open_sessions(served, 3)

    assert (await send(a, one, "a"))["status"] == "started"
    await asyncio.wait_for(gate_a.reached.wait(), 5)
    assert (await send(b, two, "b"))["status"] == "started"
    await asyncio.wait_for(gate_b.reached.wait(), 5)
    assert (await send(c, three, "c"))["status"] == "started"
    await idle(served, three)

    assert served.daemon.host(one).turn_active and served.daemon.host(two).turn_active  # both still running
    gate_a.release.set()
    gate_b.release.set()
    await idle(served, one)
    await idle(served, two)
    assert len(served.provider.requests) == 3


async def two_overlapping_turns(served: Any) -> tuple[Any, str, Any, str, Gate]:
    """A writes a.py and s.py and is held open at its second request; meanwhile B writes b.py and s.py and ends."""
    (a, one), (b, two) = await open_sessions(served, 2)
    gate = served.gate
    await send(a, one, "a")
    await asyncio.wait_for(gate.reached.wait(), 5)
    await send(b, two, "b")
    await idle(served, two)
    return a, one, b, two, gate


@needs_git
async def test_undo_skips_a_path_an_overlapping_running_turn_of_another_session_changed(serve) -> None:
    # Bug caught: B's /undo reverting s.py, which A (still running) also wrote, or touching A's a.py.
    gate = Gate()
    served = await serve(
        [writes("a", a_dot_py="A", s_dot_py="A"), [gate, *REPLY], writes("b", b_dot_py="B", s_dot_py="B"), REPLY],
        setup=allow_writes,
    )
    served.gate = gate  # type: ignore[attr-defined]
    _a, _one, b, two, _ = await two_overlapping_turns(served)

    text = await undo_text(b, two)

    ws = served.workspace
    assert not (ws / "b.py").exists()
    assert (ws / "a.py").read_text() == "A" and (ws / "s.py").read_text() == "B"
    assert "s.py (changed by another session)" in text and "1 deleted" in text
    gate.release.set()


@needs_git
async def test_undo_still_skips_the_path_after_the_other_sessions_host_was_dropped_when_idle(serve) -> None:
    # Bug caught: a session closed after its idle grace taking its ledger entries with it, so B's undo reverted s.py.
    gate = Gate()
    served = await serve(
        [writes("a", a_dot_py="A", s_dot_py="A"), [gate, *REPLY], writes("b", b_dot_py="B", s_dot_py="B"), REPLY],
        setup=allow_writes,
    )
    served.gate = gate  # type: ignore[attr-defined]
    _a, one, b, two, _ = await two_overlapping_turns(served)
    gate.release.set()
    await idle(served, one)
    daemon = served.daemon
    host = daemon.host(one)
    assert host is not None

    daemon._drop_if_unused(host, one)  # nobody watches it and it is idle: closed
    await daemon._tasks[-1]  # the close
    assert daemon.host(one) is None
    assert all(e.retired for e in daemon.ledger.entries if e.owner == host.agent.owner)  # kept only as a guard
    text = await undo_text(b, two)

    assert "s.py (changed by another session)" in text and (served.workspace / "s.py").read_text() == "B"


async def test_the_file_level_undo_skips_the_same_path_with_checkpoints_off(serve) -> None:
    # Bug caught: the fallback (no snapshots) ignoring the ledger.
    gate = Gate()
    served = await serve(
        [writes("a", a_dot_py="A", s_dot_py="A"), [gate, *REPLY], writes("b", b_dot_py="B", s_dot_py="B"), REPLY],
        setup=allow_writes,
        checkpoints=False,
    )
    served.gate = gate  # type: ignore[attr-defined]
    _a, _one, b, two, _ = await two_overlapping_turns(served)

    text = await undo_text(b, two)

    ws = served.workspace
    assert not (ws / "b.py").exists() and (ws / "a.py").read_text() == "A" and (ws / "s.py").read_text() == "B"
    assert "skipped s.py (changed by another session)" in text
    gate.release.set()


@needs_git
async def test_nothing_is_skipped_when_the_other_turn_began_after_this_one_ended(serve) -> None:
    # Bug caught: treating every other live turn as an overlap, whatever its time.
    gate = Gate()
    served = await serve(
        [writes("b", b_dot_py="B", s_dot_py="B"), REPLY, writes("a", a_dot_py="A"), [gate, *REPLY]], setup=allow_writes
    )
    (a, one), (b, two) = await open_sessions(served, 2)
    await send(b, two, "b")
    await idle(served, two)
    await send(a, one, "a")
    await asyncio.wait_for(gate.reached.wait(), 5)

    text = await undo_text(b, two)

    ws = served.workspace
    assert not (ws / "b.py").exists() and not (ws / "s.py").exists() and (ws / "a.py").read_text() == "A"
    assert "skipped" not in text and "changed by another session" not in text
    gate.release.set()


async def test_model_switch_is_still_refused_while_another_session_runs_a_turn(serve) -> None:
    # Bug caught: dropping the refusal for /model along with the one for messages.
    from core.host.session import OTHER_SESSION

    gate = Gate()
    served = await serve([[gate, *REPLY]])
    (a, one), (b, two) = await open_sessions(served, 2)
    await send(a, one, "a")
    await asyncio.wait_for(gate.reached.wait(), 5)

    refused = (await b.post(f"/v1/sessions/{two}/model", json={"name": "other:7b"})).json()

    assert refused["model"] is None and refused["notices"] == [{"level": "warn", "text": OTHER_SESSION}]
    gate.release.set()


@needs_git
async def test_diff_marks_a_path_another_session_changed_while_this_turn_ran(serve) -> None:
    # Bug caught: /diff showing a shared path as plainly this turn's, so the user does not see /undo will skip it.
    gate = Gate()
    served = await serve(
        [writes("a", a_dot_py="A", s_dot_py="A"), [gate, *REPLY], writes("b", b_dot_py="B", s_dot_py="B"), REPLY],
        setup=allow_writes,
    )
    served.gate = gate  # type: ignore[attr-defined]
    _a, _one, b, two, _ = await two_overlapping_turns(served)

    shown = (await b.get(f"/v1/sessions/{two}/diff")).json()["diff"]["items"]

    assert {i["path"]: i["foreign"] for i in shown} == {"b.py": False, "s.py": True}
    gate.release.set()


@needs_git
async def test_two_sessions_with_ten_overlapping_turns_each_undo_only_their_own_files(serve) -> None:
    # Bug caught: a corrupted shadow repo or an undo that reverts another session's file under repeated overlaps.
    rounds = 10
    gates = [Gate() for _ in range(rounds)]
    script: list[Any] = []
    for i, gate in enumerate(gates):
        script += [
            writes(f"a{i}-", **{f"a{i}_dot_py": "A", "s_dot_py": f"A{i}"}),
            [gate, *REPLY],
            writes(f"b{i}-", **{f"b{i}_dot_py": "B", "s_dot_py": f"B{i}"}),
            REPLY,
        ]
    served = await serve(script, setup=allow_writes)
    (a, one), (b, two) = await open_sessions(served, 2)
    for i, gate in enumerate(gates):
        await send(a, one, f"a{i}")
        await asyncio.wait_for(gate.reached.wait(), 5)
        await send(b, two, f"b{i}")
        await idle(served, two)
        gate.release.set()
        await idle(served, one)

    ws = served.workspace
    for _ in range(rounds):
        assert "s.py (changed by another session)" in await undo_text(a, one)
    assert sorted(p.name for p in ws.glob("*.py")) == sorted([*(f"b{i}.py" for i in range(rounds)), "s.py"])
    assert (ws / "s.py").read_text() == f"B{rounds - 1}"
    for _ in range(rounds):
        await undo_text(b, two)
    assert [p.name for p in ws.glob("b*.py")] == []

    store = served.daemon.rt.checkpoints
    assert store is not None and store.enabled, store.off_reason if store else None
    import subprocess

    subprocess.run(["git", "--git-dir", str(store.folder), "fsck", "--strict"], check=True, capture_output=True)
