"""Inputs of assemble_video: workspace audio checked once like images, and the image size limit as a parameter."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from core.engine.guardrails.setup import prepare
from core.engine.image_inputs import resolve_input
from core.engine.media import InputAudio, resolve_audio
from core.engine.tools.base import ToolContext, ToolResult
from core.testing.image_fakes import png
from core.testing.media_fakes import wav_bytes

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\0" * 64
ALL = ("wav", "mp3", "m4a")


def _context(workspace: Path, tmp_path: Path) -> ToolContext:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    guard = prepare(workspace, None, None, {}, home=home)
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=guard)


@pytest.fixture
def ctx(workspace: Path, tmp_path: Path) -> ToolContext:
    return _context(workspace, tmp_path)


def _ok(result: InputAudio | ToolResult) -> InputAudio:
    assert isinstance(result, InputAudio), result
    return result


def _refused(result: object) -> ToolResult:
    assert isinstance(result, ToolResult) and result.is_error, result
    return result


def test_a_wav_is_read_once_and_sniffed(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.wav").write_bytes(wav_bytes(1.0))
    got = _ok(resolve_audio("a.wav", ctx, ("wav",)))
    assert (got.label, got.kind, got.data) == ("a.wav", "wav", wav_bytes(1.0))
    assert got.identity == (os.stat(workspace / "a.wav").st_dev, os.stat(workspace / "a.wav").st_ino)


def test_the_kind_comes_from_the_bytes_not_the_name(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "song.wav").write_bytes(MP3)
    assert _ok(resolve_audio("song.wav", ctx, ALL)).kind == "mp3"
    assert _refused(resolve_audio("song.wav", ctx, ("wav",))).output == "song.wav is not a WAV file"


def test_a_text_file_named_wav_is_refused(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.wav").write_text("hello")
    result = _refused(resolve_audio("a.wav", ctx, ALL))
    assert result.output == "a.wav is not a WAV, MP3 or M4A file" and result.summary == "not audio"


def test_clipboard_audio_is_not_accepted(ctx: ToolContext) -> None:
    result = _refused(resolve_audio("clipboard:1", ctx, ALL))
    assert "clipboard" in result.output and result.summary == "not a file"


def test_outside_symlinked_and_protected_files_are_refused(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(wav_bytes())
    (workspace / "link.wav").symlink_to(outside)
    (workspace / ".env").write_bytes(wav_bytes())
    (workspace / "env.wav").symlink_to(workspace / ".env")
    os.link(workspace / ".env", workspace / "hard.wav")
    ctx = _context(workspace, tmp_path)  # after the files exist: the guard records protected files at launch
    for ref in (str(outside), "link.wav"):
        assert _refused(resolve_audio(ref, ctx, ALL)).output == "refused: outside workspace"
    for ref in (".env", "hard.wav"):
        assert _refused(resolve_audio(ref, ctx, ALL)).output == "protected path"
    assert _refused(resolve_audio("env.wav", ctx, ALL))


def test_a_folder_a_missing_file_and_a_fifo_are_refused(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "d.wav").mkdir()
    os.mkfifo(workspace / "f.wav")
    assert _refused(resolve_audio("d.wav", ctx, ALL)).summary == "directory"
    assert _refused(resolve_audio("none.wav", ctx, ALL)).summary == "not found"
    started = time.monotonic()
    assert "not a regular file" in _refused(resolve_audio("f.wav", ctx, ALL)).output
    assert time.monotonic() - started < 2


def test_audio_over_50_mb_is_refused_and_exactly_50_mb_is_read(workspace: Path, ctx: ToolContext) -> None:
    for name, size in (("big.wav", 50 * 1024 * 1024 + 1), ("edge.wav", 50 * 1024 * 1024)):
        with open(workspace / name, "wb") as handle:
            handle.write(wav_bytes(0.1))
            handle.truncate(size)
    result = _refused(resolve_audio("big.wav", ctx, ALL))
    assert result.summary == "audio too large" and "50 MB" in result.output
    assert len(_ok(resolve_audio("edge.wav", ctx, ALL)).data) == 50 * 1024 * 1024


def test_a_path_swapped_for_a_file_outside_the_workspace_after_the_check_is_refused(
    workspace: Path, tmp_path: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(wav_bytes())
    (workspace / "a.wav").write_bytes(wav_bytes())
    real_open = os.open
    monkeypatch.setattr(
        os, "open", lambda p, flags, *a, **k: real_open(outside if str(p).endswith("a.wav") else p, flags, *a, **k)
    )
    assert "outside workspace" in _refused(resolve_audio("a.wav", ctx, ALL)).output


def test_resolve_input_keeps_its_five_megabyte_default(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (6 * 1024 * 1024))
    result = _refused(resolve_input("big.png", ctx))
    assert result.summary == "image too large" and "limit 5 MB" in result.output


def test_resolve_input_takes_a_larger_limit(workspace: Path, ctx: ToolContext) -> None:
    limit = 20 * 1024 * 1024
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (6 * 1024 * 1024))
    got = resolve_input("big.png", ctx, max_bytes=limit)
    assert not isinstance(got, ToolResult) and (got.width, got.height) == (8, 8)
    (workspace / "huge.png").write_bytes(png(8, 8) + b"\0" * limit)
    result = _refused(resolve_input("huge.png", ctx, max_bytes=limit))
    assert result.summary == "image too large" and "limit 20 MB" in result.output
