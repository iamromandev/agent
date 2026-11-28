"""Custom commands in the daemon: the client sends `/name args`, the daemon expands it from its own loaded set."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import Gate

REPLY = [TextDelta("ok"), Done(1, 1)]


def _project_command(ws: Path) -> None:
    folder = ws / ".agent" / "commands"
    folder.mkdir(parents=True)
    (folder / "review.md").write_text("---\ndescription: Look it over\n---\nReview $ARGUMENTS\n")
    (folder / "empty.md").write_text("$ARGUMENTS")


async def _send(client, session_id: str, text: str) -> dict:
    sent = await client.post(f"/v1/sessions/{session_id}/messages", json={"text": text, "command": True})
    assert sent.status_code == 200, sent.text
    return sent.json()


async def test_a_command_is_expanded_by_the_daemon_and_shown_with_its_label(serve) -> None:
    served = await serve([REPLY], setup=_project_command)
    a, b = served.client("A"), served.client("B")
    session_id = (await served.open(a))["session_id"]
    stream = served.stream(b, session_id, after=0)
    assert (await _send(a, session_id, "/review x"))["status"] == "started"
    await stream.until(lambda s: "turn_outcome" in s.types())
    (entry,) = stream.of("user_entry")
    assert entry["text"] == "Review x" and entry["label"] == "/review (project)"
    assert served.provider.requests[0][0][-1].content == "Review x"


@pytest.mark.parametrize("raw", ["/nope", "/../x", "/a/b"])
async def test_an_unknown_name_is_refused_and_starts_no_turn(serve, raw) -> None:
    served = await serve([REPLY], setup=_project_command)
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    got = await _send(client, session_id, raw)
    assert got["status"] == "refused" and "Unknown command" in got["reason"]
    assert served.provider.requests == []


async def test_an_empty_expansion_only_warns(serve) -> None:
    served = await serve(setup=_project_command)
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    got = await _send(client, session_id, "/empty")
    assert got["status"] == "refused" and got["reason"] == "/empty is empty"


async def test_a_queued_command_that_is_interrupted_gives_back_the_typed_text(serve) -> None:
    gate = Gate()
    served = await serve([[gate, *REPLY]], setup=_project_command)
    a = served.client("A")
    session_id = (await served.open(a))["session_id"]
    stream = served.stream(a, session_id, after=0)
    await a.post(f"/v1/sessions/{session_id}/messages", json={"text": "slow"})
    await asyncio.wait_for(gate.reached.wait(), 5)
    assert (await _send(a, session_id, "/review x"))["status"] == "queued"
    await a.post(f"/v1/sessions/{session_id}/interrupt")
    await stream.until(lambda s: "turn_outcome" in s.types())
    assert stream.of("turn_outcome")[0]["restores"] == [{"client": "A", "text": "/review x"}]


async def test_the_listing_gives_name_source_and_description_escaped_and_redacted(serve) -> None:
    secret = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"

    def setup(ws: Path) -> None:
        _project_command(ws)
        (ws / ".agent" / "commands" / "leak.md").write_text(f"---\ndescription: uses {secret} \x1b[31mred\n---\nx")

    served = await serve(setup=setup)
    client = served.client()
    got = await client.get("/v1/commands")
    rows = {row["name"]: row for row in got.json()["commands"]}
    assert rows["review"] == {"name": "review", "source": "project", "description": "Look it over"}
    assert secret not in got.text and "\x1b" not in got.text


async def test_the_unknown_name_in_the_refusal_is_escaped(serve) -> None:
    served = await serve([REPLY], setup=_project_command)
    client = served.client()
    session_id = (await served.open(client))["session_id"]
    got = await _send(client, session_id, "/a\x1b[31mb\x07")
    assert got["status"] == "refused" and "\x1b" not in got["reason"] and "\x07" not in got["reason"]
    assert "Unknown command: /a" in got["reason"]


async def test_the_listing_carries_the_load_warnings_escaped_and_redacted(serve) -> None:
    secret = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"

    def setup(ws: Path) -> None:
        _project_command(ws)
        folder = ws / ".agent" / "commands"
        (folder / "model.md").write_text("built-in name")
        (folder / "bad\x1bname.md").write_text("x")
        (folder / f"{secret}.md").write_bytes(b"\xff\xfe not utf-8")

    served = await serve(setup=setup)
    got = await served.client().get("/v1/commands")
    warnings = got.json()["warnings"]
    assert len(warnings) == 3 and any("/model is a built-in command" in w for w in warnings)
    assert secret not in got.text and "\x1b" not in got.text
