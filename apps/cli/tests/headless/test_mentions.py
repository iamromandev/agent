"""`@path` mentions in headless mode: the expanded prompt, stderr notes, `--quiet`, `stream-json` notices, hooks."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.provider.base import Done, TextDelta
from src.headless import HeadlessOptions
from src.headless.run import run_headless

from tests.fakes import FakeCatalog, ScriptedProvider


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    config = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.delenv("AGENT_SEARCH_URL", raising=False)
    get_settings.cache_clear()
    yield config
    get_settings.cache_clear()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    (path / "src").mkdir(parents=True)
    (path / "src" / "a.py").write_text("one\ntwo\nthree\n")
    return path


def go(ws: Path, provider: ScriptedProvider, prompt: str, **opts: object) -> int:
    return run_headless(ws, HeadlessOptions(prompt, **opts), provider=provider)  # ty: ignore[invalid-argument-type]


def provider() -> ScriptedProvider:
    return ScriptedProvider([[TextDelta("ok"), Done(1, 1)]])


def test_a_mention_in_the_prompt_is_attached_next_to_piped_input(ws: Path) -> None:
    p = provider()

    run_headless(ws, HeadlessOptions("see @src/a.py"), provider=p, stdin=io.BytesIO(b"data @src/a.py\n"))

    sent = p.requests[0][0][-1].content
    assert sent.count("<attached ") == 1
    assert sent.endswith("<piped input>\n```\ndata @src/a.py\n```")


# --- images -----------------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2048


def vision_catalog(*, vision: bool = True, fail: bool = False) -> FakeCatalog:
    return FakeCatalog(["fake"], capabilities={"fake": ("completion", "vision") if vision else ("completion",)}, fail=fail)


def run_with_catalog(ws: Path, p: ScriptedProvider, prompt: str, catalog: FakeCatalog, **opts: object) -> int:
    return run_headless(ws, HeadlessOptions(prompt, **opts), provider=p, catalog=catalog)  # ty: ignore[invalid-argument-type]


def test_an_image_mention_carries_the_image_and_notes_it_on_stderr(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (ws / "a.png").write_bytes(PNG)
    p = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=20_000)

    code = run_with_catalog(ws, p, "see @a.png", vision_catalog())

    assert code == 0
    user = [m for m in p.requests[0][0] if m.role == "user"][-1]
    assert user.content == "see @a.png"
    assert [i.data for i in user.images] == [PNG]
    assert "attached a.png (PNG, 2 KB)" in capsys.readouterr().err


def test_without_vision_the_image_is_a_warning_and_the_text_still_goes(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (ws / "a.png").write_bytes(PNG)
    p = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=20_000)

    code = run_with_catalog(ws, p, "see @a.png", vision_catalog(vision=False))

    assert code == 0
    assert all(m.images == () for m in p.requests[0][0])
    err = capsys.readouterr().err
    assert "not attached a.png (model 'fake' has no vision)" in err
    assert "attached a.png (PNG" not in err


def test_a_failing_capability_lookup_warns_and_the_turn_runs(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (ws / "a.png").write_bytes(PNG)
    p = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=20_000)

    code = run_with_catalog(ws, p, "see @a.png", vision_catalog(fail=True))

    assert code == 0
    assert all(m.images == () for m in p.requests[0][0])
    err = capsys.readouterr().err
    assert err.count("could not read the capabilities of 'fake'") == 1


def test_stream_json_has_the_attached_image_as_a_notice_event(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    (ws / "a.png").write_bytes(PNG)
    p = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=20_000)

    code = run_with_catalog(ws, p, "see @a.png", vision_catalog(), output="stream-json")

    assert code == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    notices = [e["text"] for e in events if e["type"] == "notice"]
    assert "attached a.png (PNG, 2 KB)" in notices
    assert events[-1]["type"] == "result"
    assert set(events[-1]) == {
        "type", "result", "is_error", "exit_code", "stop_reason", "error", "session_id", "model", "duration_ms",
        "steps", "usage", "tool_calls",
    }


def test_no_image_means_no_capability_lookup(ws: Path) -> None:
    (ws / "a.txt").write_text("x\n")
    catalog = vision_catalog()
    run_with_catalog(ws, provider(), "see @a.txt", catalog)
    assert catalog.capability_calls == []
