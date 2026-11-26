"""speak: validation, dialog, execution against a fake speech helper, placement, failures."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.tools import media as tools_media
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.media import SpeakTool

from tests.helpers import assert_all_gone, wait_until
from tests.media_fakes import fake_speech, wav_bytes

TEXT = "Every morning, Mia climbed the hill."


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


@pytest.fixture(autouse=True)
def _helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setattr(tools_media, "SPEECH_HELPER", fake_speech(folder))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


@pytest.fixture
def tool(workspace: Path) -> SpeakTool:
    return SpeakTool("x/y", 20, Guardrails.off(workspace))


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))


async def _pids(file: Path) -> list[int]:
    await wait_until(lambda: file.exists() and len(file.read_text().split()) == 2)
    return [int(value) for value in file.read_text().split()]


async def _prepare(tool: SpeakTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("text", TEXT)
    args.setdefault("path", "a.wav")
    return await tool.prepare(args, ctx)


async def _refused(tool: SpeakTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


async def _prepared(tool: SpeakTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


def test_the_tool_asks_every_time_hides_in_plan_mode_and_can_be_interrupted(tool: SpeakTool) -> None:
    assert tool.name == "speak"
    assert tool.mutating and tool.needs_approval
    assert tool.runs_to_completion() is False
    assert tool.grant_arg == "path"
    assert tool.parameters["required"] == ["text", "path"]
    assert tool.label({"text": "x", "path": "o/a.wav"}) == "speak o/a.wav"
    assert tool.label({}) == "speak"


@pytest.mark.parametrize(
    "args, message",
    [
        ({"text": "   "}, "text must be 1-4000 characters"),
        ({"text": "x" * 4001}, "text must be 1-4000 characters"),
        ({"voice": "zz_nobody"}, "voice must be one of"),
        ({"speed": 0.4}, "speed must be 0.5-2.0"),
        ({"speed": 2.1}, "speed must be 0.5-2.0"),
        ({"speed": True}, "speed must be 0.5-2.0"),
        ({"path": "a.png"}, "a.png must end in .wav"),
        ({"path": "../a.wav"}, "refused: outside workspace"),
        ({"path": "/etc/a.wav"}, "refused: outside workspace"),
    ],
)
async def test_invalid_arguments_start_no_process(
    tool: SpeakTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: dict[str, Any], message: str
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_SPEECH_DUMP", str(dump))
    assert (await _refused(tool, ctx, **args)).output.startswith(message)
    assert not dump.exists()


async def test_a_lone_surrogate_is_invalid_and_starts_no_process(
    tool: SpeakTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_SPEECH_DUMP", str(dump))
    result = await _refused(tool, ctx, text="hi \ud800", path="a.wav")
    assert result.summary == "invalid" and "not valid Unicode" in result.output
    assert not dump.exists()


async def test_a_wrong_type_is_invalid(tool: SpeakTool, ctx: ToolContext) -> None:
    assert (await _refused(tool, ctx, text=5)).summary == "invalid"


async def test_a_protected_path_a_directory_and_a_symlinked_parent_are_refused(
    tool: SpeakTool, workspace: Path, tmp_path: Path
) -> None:
    class Paths:
        def is_protected(self, path: Path) -> bool:
            return path.name == "secret.wav"

    guard = dataclasses.replace(Guardrails.off(workspace), paths=Paths())  # type: ignore[arg-type]
    ctx = ToolContext(workspace, 5.0, guard=guard)
    assert (await _refused(tool, ctx, path="secret.wav")).output == "protected path"
    (workspace / "d.wav").mkdir()
    assert (await _refused(tool, ctx, path="d.wav")).output == "d.wav is a directory"
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside)
    assert (await _refused(tool, ctx, path="link/a.wav")).output == "refused: outside workspace"


async def test_the_dialog_shows_the_call(tool: SpeakTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.wav").write_bytes(b"old")
    run = await _prepared(tool, ctx, text="Hi \x1b[31mthere", voice="bm_lewis", speed=1.25)
    assert run.preview is not None
    assert run.preview.title == "speak a.wav"
    lines = run.preview.body.split("\n")
    assert lines[0] == "text: Hi \\x1b[31mthere"
    assert "voice: bm_lewis, speed 1.25" in lines
    assert "model: x/y" in lines
    assert "[overwrites the existing file]" in lines
    assert "[model x/y is not downloaded; this call may download several hundred MB]" in lines


async def test_a_cached_model_has_no_download_line(tool: SpeakTool, ctx: ToolContext, tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "w").write_text("x")
    run = await _prepared(tool, ctx)
    assert run.preview is not None
    assert "not downloaded" not in run.preview.body and "overwrites" not in run.preview.body


async def test_the_text_reaches_the_helper_only_through_a_private_file(
    tool: SpeakTool, ctx: ToolContext, tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_SPEECH_DUMP", str(dump))
    monkeypatch.setenv("GITHUB_TOKEN", "sekrit-value-1234")
    text = "$(touch pwned); `id` 'quote' \"double\" é"
    result = await (await _prepared(tool, ctx, text=f"  {text}  ", voice="af_sky", speed=1.5)).execute()
    assert not result.is_error, result
    seen = json.loads(dump.read_text())
    assert seen["text"] == text
    assert all(text not in arg and "pwned" not in arg for arg in seen["argv"])
    argv = seen["argv"]
    assert argv[argv.index("--model") + 1] == "x/y"
    assert argv[argv.index("--voice") + 1] == "af_sky"
    assert argv[argv.index("--speed") + 1] == "1.5"
    assert argv[argv.index("--text-file") + 1] == "text.txt"
    assert argv[argv.index("--out") + 1] == "speech.wav"
    assert argv[argv.index("--skipped") + 1] == "skipped.json"
    assert seen["text_mode"] == "0o600"
    assert seen["isolated"] == 1
    assert Path(seen["cwd"]).name.startswith("agent-media-")
    assert "GITHUB_TOKEN" not in seen["env"]
    assert not (workspace / "pwned").exists()


async def test_success_writes_the_wav_and_reports_it(tool: SpeakTool, ctx: ToolContext, workspace: Path) -> None:
    result = await (await _prepared(tool, ctx, path="audio/a.wav")).execute()
    path = workspace / "audio" / "a.wav"
    assert not result.is_error, result
    assert path.read_bytes() == wav_bytes(1.0)
    kb = f"{max(round(path.stat().st_size / 1024), 1)} KB"
    assert result.output == f"spoke audio/a.wav (WAV, 1.0 s, 24 kHz, {kb})"
    assert result.summary == f"WAV, 1.0 s, 24 kHz, {kb}"
    assert sorted(p.name for p in path.parent.iterdir()) == ["a.wav"]


async def test_skipped_words_are_listed_escaped_and_capped(
    tool: SpeakTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SPEECH_SKIPPED", json.dumps(["Mia", "Zorbulax"]))
    result = await (await _prepared(tool, ctx)).execute()
    assert "; skipped words: Mia, Zorbulax)" in result.output
    words = [f"w{n}" for n in range(25)]
    words[0] = "bad\x1b[0m"
    monkeypatch.setenv("FAKE_SPEECH_SKIPPED", json.dumps(words))
    result = await (await _prepared(tool, ctx)).execute()
    assert "; skipped words: bad\\x1b[0m, w1, " in result.output
    assert "w19 and 5 more)" in result.output
    assert "w20" not in result.output


async def test_an_unreadable_skipped_file_is_ignored(tool: SpeakTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SPEECH_MODE", "badskipped")
    result = await (await _prepared(tool, ctx)).execute()
    assert not result.is_error and "skipped" not in result.output


async def test_a_progress_line_is_sent(tool: SpeakTool, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    ctx = ToolContext(workspace, 5.0, guard=Guardrails.off(workspace), progress=lambda event: lines.append(event.text) if isinstance(event, ToolProgress) else None)  # type: ignore[arg-type]
    await (await _prepared(tool, ctx)).execute()
    assert lines and lines[0] == "speaking… 0:00"


async def test_success_replaces_an_existing_file_through_the_journal(tool: SpeakTool, ctx: ToolContext, workspace: Path) -> None:
    journal = Journal()
    jctx = dataclasses.replace(ctx, journal=journal)
    (workspace / "a.wav").write_bytes(b"old")
    (workspace / "a.wav").chmod(0o640)
    await (await _prepared(tool, jctx)).execute()
    target = workspace / "a.wav"
    assert target.read_bytes() == wav_bytes(1.0)
    assert target.stat().st_mode & 0o777 == 0o640
    assert journal.calls == [("before", target), ("after", target)]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("nothing", "kokoro wrote no audio"),
        ("symlink", "kokoro wrote no audio"),
        ("notwav", "kokoro wrote something that is not a WAV file"),
        ("silent", "kokoro wrote something that is not a WAV file"),
        ("fail", "kokoro failed (exit 1): "),
        ("model", "kokoro failed (exit 3): could not load the model x: OSError"),
    ],
)
async def test_failures_leave_the_old_file_alone(
    tool: SpeakTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_SPEECH_MODE", mode)
    (workspace / "a.wav").write_bytes(b"old")
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output.startswith(message)
    assert result.summary == "failed"
    assert (workspace / "a.wav").read_bytes() == b"old"
    assert [p.name for p in workspace.iterdir()] == ["a.wav"]


async def test_the_stderr_tail_is_escaped(tool: SpeakTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SPEECH_MODE", "fail")
    result = await (await _prepared(tool, ctx)).execute()
    assert "\x1b" not in result.output and "\\x1b[2J" in result.output


async def test_a_huge_output_is_refused(tool: SpeakTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_media, "MAX_AUDIO_BYTES", 1000)
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output == "kokoro wrote audio over 50 MB"


async def test_a_timeout_is_a_timed_out_error(
    workspace: Path, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SPEECH_MODE", "sleep")
    monkeypatch.setenv("FAKE_SPEECH_PIDS", str(tmp_path / "pids"))
    slow = SpeakTool("x/y", 1, Guardrails.off(workspace))
    (workspace / "a.wav").write_bytes(b"old")
    result = await (await _prepared(slow, ctx)).execute()
    assert result.output == "timed out after 1 s" and result.summary == "timed out"
    assert (workspace / "a.wav").read_bytes() == b"old"
    await assert_all_gone(*await _pids(tmp_path / "pids"))


async def test_a_cancel_kills_the_process_group(
    tool: SpeakTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_SPEECH_MODE", "sleep")
    monkeypatch.setenv("FAKE_SPEECH_PIDS", str(pids))
    task = asyncio.ensure_future((await _prepared(tool, ctx)).execute())
    started = await _pids(pids)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await assert_all_gone(*started)
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("mode", ["ok", "fail", "nothing", "notwav", "token"])
async def test_the_temp_folder_is_removed_after_every_outcome(
    tool: SpeakTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv("FAKE_SPEECH_MODE", mode)
    await (await _prepared(tool, ctx)).execute()
    assert list(temp.iterdir()) == []


async def test_a_folder_swapped_for_a_symlink_during_the_run_is_refused(
    tool: SpeakTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    run = await _prepared(tool, ctx, path="sub/a.wav")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "sub").symlink_to(outside)
    result = await run.execute()
    assert result.is_error and result.output == "refused: outside workspace"
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
async def test_an_unwritable_folder_is_a_cannot_write_error(tool: SpeakTool, ctx: ToolContext, workspace: Path) -> None:
    folder = workspace / "ro"
    folder.mkdir()
    run = await _prepared(tool, ctx, path="ro/a.wav")
    folder.chmod(0o500)
    try:
        result = await run.execute()
    finally:
        folder.chmod(0o700)
    assert result.is_error and result.output.startswith("Cannot write ro/a.wav: ")
    assert result.summary == "write failed"


async def test_a_missing_helper_is_an_error(tool: SpeakTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = await _prepared(tool, ctx)
    monkeypatch.setattr(tools_media, "SPEECH_HELPER", tmp_path / "gone.py")
    result = await run.execute()
    assert result.is_error and result.output == "the speech helper is missing"
