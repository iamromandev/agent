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

from tests.fakes import ScriptedProvider


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
