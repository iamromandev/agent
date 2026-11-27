"""The snapshot a client attaches with: image bytes stay in the daemon and every text is redacted."""

from __future__ import annotations

import json
from pathlib import Path

from core.engine import Agent
from core.engine.approval import Approver
from core.engine.provider.base import Image, Message
from core.engine.tools.base import ToolContext
from core.host.session import SessionHost
from core.server.snapshot import snapshot
from core.testing.fakes import ScriptedProvider


def mask(text: str) -> str:
    return text.replace("hunter2hunter2", "[redacted:env:X]")


def test_records_carry_image_metadata_only_and_redacted_text(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    provider = ScriptedProvider([])

    def factory(approver: Approver) -> Agent:
        return Agent(provider, [], approver, ToolContext(root, 5.0), 5)

    host = SessionHost(factory, provider, root, window_cap=8192)
    host.agent.history.append(Message("user", "look hunter2hunter2", images=(Image("png", b"PIXELBYTES", "a.png"),)))
    host.agent.history.append(Message("assistant", "seen hunter2hunter2"))

    data = snapshot(host, seq=3, redact=mask).model_dump(mode="json")
    text = json.dumps(data)

    assert "PIXELBYTES" not in text and "UElYRUxCWVRFUw" not in text  # neither raw nor base64
    assert "hunter2hunter2" not in text
    user, reply = data["records"]
    assert user["images"] == [{"format": "png", "name": "a.png", "size": 10}]
    assert user["content"] == "look [redacted:env:X]" and reply["content"] == "seen [redacted:env:X]"
    assert data["seq"] == 3 and data["claims"]["turn_active"] is False


def test_a_client_draws_the_snapshot_records_as_the_store_would_with_image_lines(tmp_path: Path) -> None:
    from core.engine.session.store import TranscriptItem
    from core.host.replay import snapshot_items

    root = tmp_path.resolve()
    provider = ScriptedProvider([])

    def factory(approver: Approver) -> Agent:
        return Agent(provider, [], approver, ToolContext(root, 5.0), 5)

    host = SessionHost(factory, provider, root, window_cap=8192)
    host.agent.history.append(Message("user", "look", images=(Image("png", b"PIXELBYTES", "a.png"),)))
    host.agent.history.append(Message("assistant", "seen"))
    records = snapshot(host, seq=0, redact=mask).model_dump(mode="json")["records"]

    assert snapshot_items(records) == [
        TranscriptItem("user", "look"),
        TranscriptItem("notice", "image a.png (PNG, 1 KB)"),
        TranscriptItem("assistant", "seen"),
    ]
