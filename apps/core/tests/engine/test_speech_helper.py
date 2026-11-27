"""The speech helper's pure parts (it never runs here: the engine is an optional extra)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from core.engine.media import SPEECH_HELPER


def load_helper() -> ModuleType:
    assert SPEECH_HELPER is not None
    spec = importlib.util.spec_from_file_location("speech_helper_under_test", SPEECH_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Token:
    text: str
    phonemes: str | None
    whitespace: str = " "


def test_importing_the_helper_needs_none_of_the_extras() -> None:
    before = {name for name in ("mlx_audio", "misaki", "numpy") if name in sys.modules}
    load_helper()
    assert {name for name in ("mlx_audio", "misaki", "numpy") if name in sys.modules} == before


def test_skipped_words_are_the_tokens_without_phonemes() -> None:
    helper = load_helper()
    tokens = [
        Token("Every", "EvRi"),
        Token("morning", "mORniN"),
        Token(",", ",", ""),
        Token("Mia", None),
        Token("climbed", "klImd"),
        Token("Zorbulax", None),
        Token("Qux", "❓"),
        Token(" ", None, ""),
        Token("\n", None, ""),
        Token("Mia", None),
    ]
    assert helper.skipped_words(tokens, "❓") == ["Mia", "Zorbulax", "Qux"]


def test_no_skipped_words() -> None:
    assert load_helper().skipped_words([Token("hi", "hI")], "❓") == []


def test_silent_symbols_with_empty_phonemes_are_not_skipped() -> None:
    assert load_helper().skipped_words([Token("$", ""), Token("#", "")], "❓") == []


def test_a_partly_unknown_word_is_skipped() -> None:
    tokens = [Token("3:30pm", "tri❓ pi")]
    assert load_helper().skipped_words(tokens, "❓") == ["3:30pm"]


def test_real_g2p_reports_only_unpronounceable_words() -> None:
    en = pytest.importorskip("misaki.en")
    pytest.importorskip("en_core_web_sm")
    helper = load_helper()
    g2p = en.G2P(trf=False, british=False, fallback=None, unk=helper.UNKNOWN)
    _, tokens = g2p("Mia costs $5 at 3:30pm, #1.")
    assert helper.skipped_words(tokens, helper.UNKNOWN) == ["Mia", "3:30pm"]


def test_a_failing_skipped_step_writes_an_empty_list_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    helper = load_helper()

    def boom(name: str) -> None:
        raise RuntimeError("no g2p")

    monkeypatch.setattr(helper.importlib, "import_module", boom)
    out = tmp_path / "skipped.json"
    helper.write_skipped(str(out), "hello", "a")
    assert out.read_text() == "[]"
    assert "skipped words" in capsys.readouterr().err


def test_the_helper_refuses_missing_arguments_with_exit_2() -> None:
    assert SPEECH_HELPER is not None
    done = subprocess.run([sys.executable, "-I", str(SPEECH_HELPER)], capture_output=True)
    assert done.returncode == 2


def test_the_espeak_stub_fails_the_fallback_like_the_broken_library() -> None:
    helper = load_helper()
    saved = sys.modules.get("misaki.espeak")
    try:
        helper.install_espeak_stub()
        stub = sys.modules["misaki.espeak"]
        with pytest.raises(RuntimeError):
            stub.EspeakFallback()
    finally:
        if saved is None:
            sys.modules.pop("misaki.espeak", None)
        else:
            sys.modules["misaki.espeak"] = saved
