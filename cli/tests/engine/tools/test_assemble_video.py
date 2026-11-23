"""assemble_video: validation, dialog, execution against a fake ffmpeg and a fake media helper, placement, failures."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.media import format_text
from src.engine.provider.base import Image
from src.engine.tools import media as tools_media
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.media import AssembleVideoTool

from tests.helpers import assert_all_gone, wait_until
from tests.image_fakes import png
from tests.media_fakes import fake_ffmpeg, fake_media, mp4_bytes, wav_bytes

HOSTILE = "it's 100% \\n done; drawtext=x %{pts}"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\0" * 64


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


@pytest.fixture(autouse=True)
def _fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setattr(tools_media, "MEDIA_HELPER", fake_media(folder))
    monkeypatch.setenv("FAKE_FFMPEG_DUMP", str(tmp_path / "ffmpeg.jsonl"))
    monkeypatch.setenv("FAKE_MEDIA_DUMP", str(tmp_path / "media.jsonl"))
    (folder / "temp").mkdir()
    monkeypatch.setenv("TMPDIR", str(folder / "temp"))
    monkeypatch.setattr("tempfile.tempdir", None)


@pytest.fixture
def tool(tmp_path: Path, workspace: Path) -> AssembleVideoTool:
    return AssembleVideoTool(fake_ffmpeg(tmp_path / "bin"), 20, Guardrails.off(workspace))


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))


@pytest.fixture(autouse=True)
def files(workspace: Path) -> None:
    (workspace / "a.png").write_bytes(png(40, 30))
    (workspace / "b.png").write_bytes(png(30, 40))
    (workspace / "n1.wav").write_bytes(wav_bytes(2.8))
    (workspace / "n2.wav").write_bytes(wav_bytes(4.3))
    (workspace / "m.mp3").write_bytes(MP3)


def scenes(*items: dict[str, Any]) -> list[dict[str, Any]]:
    return list(items)


TWO = scenes({"image": "a.png", "audio": "n1.wav"}, {"image": "b.png", "audio": "n2.wav"})


async def _prepare(tool: AssembleVideoTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("scenes", TWO)
    args.setdefault("path", "out.mp4")
    return await tool.prepare(args, ctx)


async def _refused(tool: AssembleVideoTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


async def _prepared(tool: AssembleVideoTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


def _runs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_the_tool_asks_every_time_hides_in_plan_mode_and_can_be_interrupted(tool: AssembleVideoTool) -> None:
    assert tool.name == "assemble_video"
    assert tool.mutating and tool.needs_approval
    assert tool.runs_to_completion() is False
    assert tool.grant_arg == "path"
    assert tool.parameters["required"] == ["scenes", "path"]
    assert tool.label({"path": "o/a.mp4"}) == "assemble_video o/a.mp4"
    assert tool.label({}) == "assemble_video"


@pytest.mark.parametrize(
    "args, message",
    [
        ({"scenes": []}, "scenes must have 1-30 items"),
        ({"scenes": "x"}, '"scenes" must be an array'),
        ({"scenes": ["a.png"]}, "scene 1 must be an object"),
        ({"scenes": [{"audio": "n1.wav"}]}, "scene 1 needs an image"),
        ({"scenes": [{"image": "a.png"}]}, "scene 1 needs audio or a duration"),
        ({"scenes": [{"image": "a.png", "duration": 0.5}]}, "scene 1 duration must be 1-60 s"),
        ({"scenes": [{"image": "a.png", "duration": 61}]}, "scene 1 duration must be 1-60 s"),
        ({"scenes": [{"image": "a.png", "duration": True}]}, "scene 1 duration must be 1-60 s"),
        ({"scenes": [{"image": "a.png", "duration": 2, "motion": "spin"}]}, "scene 1 motion must be one of"),
        ({"scenes": [{"image": "a.png", "duration": 2, "text": "x" * 201}]}, "scene 1 text must be at most 200 characters"),
        ({"scenes": [{"image": "a.png", "duration": 2, "text": "word " * 30}]}, "scene 1 text needs more than 4 lines"),
        ({"scenes": [{"image": "a.png", "duration": 2, "text": 5}]}, "scene 1 text must be a string"),
        ({"crossfade": 1.5}, "crossfade must be 0-1.0"),
        ({"crossfade": -0.1}, "crossfade must be 0-1.0"),
        ({"path": "out.png"}, "out.png must end in .mp4"),
        ({"path": "../out.mp4"}, "refused: outside workspace"),
        ({"scenes": [{"image": "a.png", "audio": "m.mp3"}]}, "m.mp3 is not a WAV file"),
        ({"scenes": [{"image": "a.png", "audio": "clipboard:1"}]}, "clipboard"),
        ({"scenes": [{"image": "a.png", "duration": 2}], "music": "a.png"}, "a.png is not a WAV, MP3 or M4A file"),
        ({"scenes": [{"image": "a.png", "audio": "n1.wav", "duration": 2}] * 31}, "scenes must have 1-30 items"),
    ],
)
async def test_invalid_arguments_start_no_process(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path, args: dict[str, Any], message: str
) -> None:
    assert (await _refused(tool, ctx, **args)).output.startswith(message)
    assert not (tmp_path / "ffmpeg.jsonl").exists() and not (tmp_path / "media.jsonl").exists()


async def test_audio_that_is_not_a_readable_wav_is_invalid(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\0" * 40)
    result = await _refused(tool, ctx, scenes=[{"image": "a.png", "audio": "bad.wav"}])
    assert result.output == "scene 1 audio bad.wav is not a PCM WAV file with sound" and result.summary == "invalid"


async def test_a_scene_over_60_seconds_or_under_1_is_invalid(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "long.wav").write_bytes(wav_bytes(61, rate=8000))
    (workspace / "short.wav").write_bytes(wav_bytes(0.5))
    assert (await _refused(tool, ctx, scenes=[{"image": "a.png", "audio": "long.wav"}])).output == (
        "scene 1 is 61.0 s; at most 60 s"
    )
    assert (await _refused(tool, ctx, scenes=[{"image": "a.png", "audio": "short.wav"}])).output == (
        "scene 1 is 0.5 s; at least 1 s"
    )


async def test_a_video_over_180_seconds_is_refused_before_any_process(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path
) -> None:
    four = scenes(*[{"image": "a.png", "duration": 60}] * 4)
    result = await _refused(tool, ctx, scenes=four)
    assert result.output == "the video would be 241.5 s; at most 180 s" and result.summary == "invalid"
    assert not (tmp_path / "media.jsonl").exists()
    three = await _prepare(tool, ctx, scenes=scenes(*[{"image": "a.png", "duration": 60}] * 3), crossfade=0)
    assert isinstance(three, Prepared)  # exactly 180 s is allowed


async def test_30_scenes_are_accepted(tool: AssembleVideoTool, ctx: ToolContext) -> None:
    run = await _prepare(tool, ctx, scenes=scenes(*[{"image": "a.png", "duration": 1}] * 30))
    assert isinstance(run, Prepared)


async def test_31_scenes_are_refused(tool: AssembleVideoTool, ctx: ToolContext) -> None:
    result = await _refused(tool, ctx, scenes=scenes(*[{"image": "a.png", "duration": 1}] * 31))
    assert result.output == "scenes must have 1-30 items"


async def test_image_refusals(tool: AssembleVideoTool, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(8, 8))
    (workspace / "link.png").symlink_to(outside)
    (workspace / ".env").write_bytes(png(8, 8))
    (workspace / "text.png").write_text("just text")
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (21 * 1024 * 1024))
    (workspace / "huge.png").write_bytes(_png_header(8000, 7000))
    ctx = ToolContext(workspace, 5.0, guard=_guard(workspace, tmp_path))
    expected = {
        str(outside): "refused: outside workspace",
        "link.png": "refused: outside workspace",
        ".env": "protected path",
        "text.png": "text.png is not an image (PNG, JPEG, GIF or WebP)",
        "big.png": "big.png: image too large (",
        "huge.png": "scene 1 image huge.png is 8000\N{MULTIPLICATION SIGN}7000: over 50 megapixels",
        "none.png": "No such file: none.png",
        "clipboard:1": "clipboard:1: no such paste",
    }
    for ref, message in expected.items():
        result = await _refused(tool, ctx, scenes=[{"image": ref, "duration": 2}])
        assert result.output.startswith(message), (ref, result.output)
    assert "limit 20 MB" in (await _refused(tool, ctx, scenes=[{"image": "big.png", "duration": 2}])).output


def _guard(workspace: Path, tmp_path: Path) -> Guardrails:
    from src.engine.guardrails.setup import prepare

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return prepare(workspace, None, None, {}, "Linux", home=home)


async def test_a_directory_target_and_a_symlinked_parent_are_refused(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    (workspace / "d.mp4").mkdir()
    assert (await _refused(tool, ctx, path="d.mp4")).output == "d.mp4 is a directory"
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside)
    assert (await _refused(tool, ctx, path="link/out.mp4")).output == "refused: outside workspace"


async def test_a_pasted_image_can_be_a_scene(tool: AssembleVideoTool, workspace: Path) -> None:
    ctx = ToolContext(workspace, 5.0, guard=Guardrails.off(workspace), pasted=(Image("png", png(8, 8), "clipboard"),))
    result = await (await _prepared(tool, ctx, scenes=[{"image": "clipboard:1", "duration": 2}])).execute()
    assert not result.is_error, result


async def test_the_dialog_shows_every_scene_the_music_the_fade_and_the_total(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "out.mp4").write_bytes(b"old")
    run = await _prepared(
        tool,
        ctx,
        scenes=[
            {"image": "a.png", "audio": "n1.wav", "motion": "pan-left", "text": "Hi \x1b[31m" + "x" * 80},
            {"image": "b.png", "duration": 5},
        ],
        music="m.mp3",
        crossfade=0.25,
    )
    assert run.preview is not None and run.preview.title == "assemble_video out.mp4"
    lines = run.preview.body.split("\n")
    assert lines[0] == "scene 1: a.png, narration n1.wav (2.8 s), pan-left, text: Hi \\x1b[31m" + "x" * (60 - len("Hi \x1b[31m"))  # cut to 60 characters first, then escaped
    assert lines[1] == "scene 2: b.png, 5.0 s, zoom-in"
    assert lines[2:] == ["music: m.mp3", "crossfade: 0.25 s", "total: 8.1 s", "[overwrites the existing file]"]


async def test_the_dialog_without_music_or_an_existing_file(tool: AssembleVideoTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx)
    assert run.preview is not None
    assert run.preview.body.split("\n")[2:] == ["music: none", "crossfade: 0.5 s", "total: 7.6 s"]


async def test_success_places_the_mp4_and_reports_it(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path) -> None:
    result = await (await _prepared(tool, ctx, path="video/out.mp4")).execute()
    path = workspace / "video" / "out.mp4"
    assert not result.is_error, result
    kb = f"{max(round(path.stat().st_size / 1024), 1)} KB"
    assert result.output == f"assembled video/out.mp4 (MP4, 1080\N{MULTIPLICATION SIGN}1920, 7.6 s, 2 scenes, {kb})"
    assert result.summary == f"MP4, 1080\N{MULTIPLICATION SIGN}1920, 7.6 s, 2 scenes, {kb}"
    assert sorted(p.name for p in path.parent.iterdir()) == ["out.mp4"]


async def test_the_runs_are_the_frames_then_the_scenes_then_the_join(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path
) -> None:
    run = await _prepared(tool, ctx, scenes=[{"image": "a.png", "audio": "n1.wav", "text": "Hello"}, {"image": "b.png", "audio": "n2.wav"}], music="m.mp3")
    assert not (await run.execute()).is_error
    steps = [(r["step"], r["argv"][0]) for r in _runs(tmp_path / "media.jsonl")]
    assert steps == [("frame", "frame"), ("caption", "caption"), ("frame", "frame")]
    kinds = [r["kind"] for r in _runs(tmp_path / "ffmpeg.jsonl")]
    assert kinds == ["scene", "scene", "final"]
    first = _runs(tmp_path / "media.jsonl")[0]
    argv = first["argv"]
    assert argv[argv.index("--in") + 1] == "in-0.png" and argv[argv.index("--out") + 1] == "frame-0.png"
    assert (argv[argv.index("--width") + 1], argv[argv.index("--height") + 1]) == ("1080", "1920")
    assert first["isolated"] == 1 and first["in_mode"] == "0o600"
    assert Path(first["cwd"]).name.startswith("agent-media-")


async def test_the_example_puts_the_second_narration_at_3_3_seconds_and_ends_at_7_6(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path
) -> None:
    assert not (await (await _prepared(tool, ctx)).execute()).is_error
    final = _runs(tmp_path / "ffmpeg.jsonl")[-1]
    argv = final["argv"]
    graph = argv[argv.index("-filter_complex") + 1]
    assert "adelay=3300:all=1" in graph and "offset=2.800" in graph
    assert argv[argv.index("-t") + 1] == "7.600" and argv[-1] == "out.mp4"
    assert final["inputs"] == {"clip-0.mp4": True, "clip-1.mp4": True, "nar-0.wav": True, "nar-1.wav": True}
    scene_lengths = [r["argv"][r["argv"].index("-t") + 1] for r in _runs(tmp_path / "ffmpeg.jsonl")[:2]]
    assert scene_lengths == ["3.300", "4.800"]


async def test_the_music_is_staged_under_its_kind_and_looped(tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path) -> None:
    assert not (await (await _prepared(tool, ctx, music="m.mp3")).execute()).is_error
    final = _runs(tmp_path / "ffmpeg.jsonl")[-1]
    assert final["inputs"]["music.mp3"] is True and "-stream_loop" in final["argv"]


async def test_the_same_image_twice_is_accepted(tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path) -> None:
    result = await (await _prepared(tool, ctx, scenes=[{"image": "a.png", "duration": 2}, {"image": "a.png", "duration": 3}])).execute()
    assert not result.is_error, result
    assert [r["argv"][r["argv"].index("--in") + 1] for r in _runs(tmp_path / "media.jsonl")] == ["in-0.png", "in-1.png"]


async def test_a_hostile_caption_reaches_only_the_caption_file(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path, workspace: Path
) -> None:
    result = await (await _prepared(tool, ctx, scenes=[{"image": "a.png", "duration": 2, "text": HOSTILE}])).execute()
    assert not result.is_error, result
    caption = next(r for r in _runs(tmp_path / "media.jsonl") if r["step"] == "caption")
    wrapped = format_text(HOSTILE)
    assert caption["text"] == wrapped and wrapped is not None and "drawtext" in wrapped
    assert caption["in_mode"] == "0o600"
    argv = caption["argv"]
    assert argv[argv.index("--text-file") + 1] == "cap-0.txt"
    everything = json.dumps([r["argv"] for r in _runs(tmp_path / "media.jsonl") + _runs(tmp_path / "ffmpeg.jsonl")])
    for piece in ("drawtext", "%{pts}", "100%", "it's"):
        assert piece not in everything
    assert result.output.count("drawtext") == 0
    assert [p.name for p in workspace.iterdir() if "drawtext" in p.name] == []


async def test_a_progress_line_is_sent_for_each_stage(tool: AssembleVideoTool, workspace: Path) -> None:
    lines: list[str] = []
    ctx = ToolContext(workspace, 5.0, guard=Guardrails.off(workspace), progress=lambda event: lines.append(event.text) if isinstance(event, ToolProgress) else None)  # type: ignore[arg-type]
    await (await _prepared(tool, ctx)).execute()
    assert lines == ["framing 1/2", "framing 2/2", "scene 1/2", "scene 2/2", "joining…"]


async def test_success_replaces_an_existing_file_through_the_journal(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path) -> None:
    journal = Journal()
    jctx = dataclasses.replace(ctx, journal=journal)
    (workspace / "out.mp4").write_bytes(b"old")
    (workspace / "out.mp4").chmod(0o640)
    await (await _prepared(tool, jctx)).execute()
    target = workspace / "out.mp4"
    assert target.read_bytes() != b"old" and target.stat().st_mode & 0o777 == 0o640
    assert journal.calls == [("before", target), ("after", target)]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("wrongsize", "ffmpeg wrote 720\N{MULTIPLICATION SIGN}1280, expected 1080\N{MULTIPLICATION SIGN}1920"),
        ("wronglen", "ffmpeg wrote 12.6 s of video, expected 7.6 s"),
        ("shortvideo", "ffmpeg wrote 6.6 s of video, expected 7.6 s"),
        ("notmp4", "ffmpeg wrote something that is not an MP4 file"),
        ("slowstart", "ffmpeg wrote an MP4 that is not faststart"),
        ("nothing", "ffmpeg wrote no video"),
        ("fail", "ffmpeg failed (exit 1): "),
    ],
)
async def test_a_bad_output_is_an_error_and_the_target_is_untouched(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_FFMPEG_MODE", mode)
    monkeypatch.setenv("FAKE_FFMPEG_ONLY", "final")
    (workspace / "out.mp4").write_bytes(b"old")
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output.startswith(message), result.output
    assert (workspace / "out.mp4").read_bytes() == b"old"
    assert sorted(p.name for p in workspace.iterdir() if p.name.endswith((".mp4", ".tmp"))) == ["out.mp4"]


async def test_a_scene_run_that_writes_nothing_is_an_error(tool: AssembleVideoTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "nothing")
    monkeypatch.setenv("FAKE_FFMPEG_ONLY", "scene")
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output == "ffmpeg wrote no video"


async def test_an_output_over_the_size_cap_is_refused(tool: AssembleVideoTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_media, "MAX_VIDEO_BYTES", 100)
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output == "ffmpeg wrote a video over 200 MB"


async def test_only_the_start_of_the_output_is_read_and_moov_past_it_is_not_faststart(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = len(mp4_bytes(1080, 1920, 7.6))
    monkeypatch.setattr(tools_media, "HEADER_BYTES", whole - 40, raising=False)  # the mdat is cut, the moov is not
    run = await _prepared(tool, ctx)
    assert not (await run.execute()).is_error
    monkeypatch.setattr(tools_media, "HEADER_BYTES", whole - 100, raising=False)  # now the moov is cut
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output == "ffmpeg wrote an MP4 that is not faststart"


async def test_the_output_is_opened_without_following_a_symlink_and_sized_from_the_same_descriptor(
    tool: AssembleVideoTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[int] = []
    real = tools_media.os.open

    def spy(path: Any, flags: int, *rest: Any, **kw: Any) -> int:
        if Path(path).name == "out.mp4":
            opened.append(flags)
        return real(path, flags, *rest, **kw)

    monkeypatch.setattr(tools_media.os, "open", spy)
    assert not (await (await _prepared(tool, ctx)).execute()).is_error
    assert opened and all(flags & tools_media.os.O_NOFOLLOW for flags in opened)


async def test_inputs_over_300_mb_are_refused_before_the_dialog(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (19 * 1024 * 1024))
    sixteen = scenes(*[{"image": "big.png", "duration": 2}] * 16)  # staged once per scene: 16 x 19 MB
    result = await _refused(tool, ctx, scenes=sixteen)
    assert result.output == "the inputs total 304 MB; at most 300 MB" and result.summary == "invalid"
    assert not (tmp_path / "media.jsonl").exists() and not (tmp_path / "ffmpeg.jsonl").exists()
    assert isinstance(await _prepare(tool, ctx, scenes=sixteen[:15]), Prepared)


async def test_music_counts_toward_the_input_budget(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (19 * 1024 * 1024))
    (workspace / "big.wav").write_bytes(wav_bytes(1.0) + b"\0" * (30 * 1024 * 1024))
    fifteen = scenes(*[{"image": "big.png", "duration": 2}] * 15)  # 285 MB, then 30 MB of music
    result = await _refused(tool, ctx, scenes=fifteen, music="big.wav")
    assert result.output == "the inputs total 315 MB; at most 300 MB"


async def test_the_ffmpeg_stderr_tail_is_escaped(tool: AssembleVideoTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "fail")
    result = await (await _prepared(tool, ctx)).execute()
    assert "\x1b" not in result.output and "\\x1b[2J" in result.output and result.summary == "failed"


@pytest.mark.parametrize(
    "mode, message",
    [
        ("fail", "media helper failed (exit 1): helper broke"),
        ("undecodable", "an image could not be decoded"),
        ("toolarge", "image too large for a video (limit 50 megapixels)"),
        ("nothing", "media helper wrote no image"),
    ],
)
async def test_media_helper_failures(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_MEDIA_MODE", mode)
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output.startswith(message), result.output
    assert "\x1b" not in result.output
    assert _runs(tmp_path / "ffmpeg.jsonl") == []


async def test_a_timeout_kills_the_group_and_leaves_the_target(
    workspace: Path, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "sleep")
    monkeypatch.setenv("FAKE_FFMPEG_PIDS", str(tmp_path / "pids"))
    slow = AssembleVideoTool(fake_ffmpeg(tmp_path / "bin"), 1, Guardrails.off(workspace))
    (workspace / "out.mp4").write_bytes(b"old")
    result = await (await _prepared(slow, ctx)).execute()
    assert result.output == "timed out after 1 s" and result.summary == "timed out"
    assert (workspace / "out.mp4").read_bytes() == b"old"
    await wait_until(lambda: (tmp_path / "pids").exists() and len((tmp_path / "pids").read_text().split()) == 2)
    await assert_all_gone(*(int(v) for v in (tmp_path / "pids").read_text().split()))


async def test_one_deadline_covers_every_run(
    workspace: Path, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MEDIA_DELAY", "0.7")
    slow = AssembleVideoTool(fake_ffmpeg(tmp_path / "bin"), 1.2, Guardrails.off(workspace))
    # each frame run alone (0.7 s) is under 1.2 s; only a deadline shared by the two runs runs out
    result = await (await _prepared(slow, ctx)).execute()
    assert result.output == "timed out after 1.2 s", result.output


async def test_a_cancel_kills_the_process_group(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "sleep")
    monkeypatch.setenv("FAKE_FFMPEG_PIDS", str(pids))
    task = asyncio.ensure_future((await _prepared(tool, ctx)).execute())
    await wait_until(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await assert_all_gone(*(int(v) for v in pids.read_text().split()))
    assert not (workspace / "out.mp4").exists()
    assert list((tmp_path / "bin" / "temp").iterdir()) == []


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"FAKE_FFMPEG_MODE": "fail"},
        {"FAKE_FFMPEG_MODE": "wrongsize"},
        {"FAKE_FFMPEG_MODE": "nothing"},
        {"FAKE_MEDIA_MODE": "fail"},
        {"FAKE_MEDIA_MODE": "toolarge"},
    ],
)
async def test_the_private_folder_is_removed_after_every_outcome(
    tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    await (await _prepared(tool, ctx)).execute()
    assert list((tmp_path / "bin" / "temp").iterdir()) == []


async def test_a_folder_swapped_for_a_symlink_during_the_run_is_refused(
    tool: AssembleVideoTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    run = await _prepared(tool, ctx, path="sub/out.mp4")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "sub").symlink_to(outside)
    result = await run.execute()
    assert result.is_error and result.output == "refused: outside workspace"
    assert list(outside.iterdir()) == []


async def test_a_target_that_cannot_be_written_is_a_cannot_write_error(tool: AssembleVideoTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "d").write_text("a file where a folder should be")
    result = await (await _prepared(tool, ctx, path="d/out.mp4")).execute()
    assert result.is_error and result.output.startswith("Cannot write d/out.mp4: ") and result.summary == "write failed"


async def test_a_missing_helper_is_an_error(tool: AssembleVideoTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = await _prepared(tool, ctx)
    monkeypatch.setattr(tools_media, "MEDIA_HELPER", tmp_path / "gone.py")
    result = await run.execute()
    assert result.is_error and result.output == "the media helper is missing"
