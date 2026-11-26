"""edit_image: validation, input checks, preview and execution against a fake mflux (AC2, AC3, AC5-AC7, AC9-AC11, AC13)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import EDIT_COMMAND, EDIT_EXTRA, MfluxGenerator, png_size
from src.engine.provider.base import Image
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.image import EditImageTool

from tests.engine.test_image_inputs import jpeg, webp_lossy
from tests.image_fakes import fake_mflux, png


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


@pytest.fixture
def make_tool(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    command = fake_mflux(folder, EDIT_COMMAND)

    def make(model: str = "x/y", timeout: float = 20) -> EditImageTool:
        return EditImageTool(MfluxGenerator(command, model, timeout, Guardrails.off(workspace), extra=EDIT_EXTRA))

    return make


@pytest.fixture
def tool(make_tool: Any) -> EditImageTool:
    return make_tool()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    (workspace / "src.png").write_bytes(png(64, 48))
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))


async def _prepare(tool: EditImageTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("prompt", "make the sky red")
    args.setdefault("image", "src.png")
    args.setdefault("path", "out.png")
    return await tool.prepare(args, ctx)


async def _refused(tool: EditImageTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


async def _prepared(tool: EditImageTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


def test_the_tool_asks_every_time_hides_in_plan_mode_and_can_be_interrupted(tool: EditImageTool) -> None:
    assert tool.name == "edit_image"
    assert tool.mutating and tool.needs_approval
    assert tool.runs_to_completion() is False
    assert tool.grant_arg == "prompt"
    assert tool.parameters["properties"]["references"]["maxItems"] == 3


async def test_bad_arguments_are_invalid_errors_naming_the_argument(tool: EditImageTool, ctx: ToolContext) -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"prompt": ""}, "prompt"),
        ({"prompt": "x" * 2001}, "prompt"),
        ({"references": ["a.png", "b.png", "c.png", "d.png"]}, "references"),
        ({"references": [1]}, "references"),
        ({"width": 100}, "width"),
        ({"width": 1000}, "width"),
        ({"height": 2064}, "height"),
        ({"steps": 0}, "steps"),
        ({"steps": 51}, "steps"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"path": "a.jpg"}, "a.jpg"),
        ({"width": "big"}, "width"),
        ({"image": None}, "image"),
    ]
    for args, name in cases:
        result = await _refused(tool, ctx, **args)
        assert result.summary == "invalid" and name in result.output, (args, result)


async def test_the_default_size_is_the_sources_rounded_to_16(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "wide.png").write_bytes(png(1000, 750))
    run = await _prepared(tool, ctx, image="wide.png")
    assert run.preview is not None and "size: 1008\u00d7752, 4 steps, seed " in run.preview.body
    result = await run.execute()
    assert png_size((workspace / "out.png").read_bytes()) == (1008, 752), result


async def test_a_large_source_defaults_to_a_scaled_size_and_one_given_side_uses_the_same_default(
    tool: EditImageTool, ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "big.png").write_bytes(png(4000, 3000))
    run = await _prepared(tool, ctx, image="big.png")
    assert run.preview is not None and "size: 1024\u00d7768, " in run.preview.body
    run = await _prepared(tool, ctx, image="big.png", width=2048)
    assert run.preview is not None and "size: 2048\u00d7768, " in run.preview.body


async def test_refused_inputs_never_reach_a_dialog(tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "o.png"
    outside.write_bytes(png(8, 8))
    (workspace / "link.png").symlink_to(outside)
    (workspace / ".env").write_text("A=1\n")
    (workspace / "x.png").write_text("text")
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (6 * 1024 * 1024))
    (workspace / "dir").mkdir()
    one = dataclasses.replace(ctx, pasted=(Image("png", png(8, 8), "clipboard"),))
    cases = [
        (ctx, {"image": str(outside)}, "refused: outside workspace"),
        (ctx, {"image": "link.png"}, "refused: outside workspace"),
        (ctx, {"image": ".env"}, "refused: protected path"),
        (ctx, {"image": "x.png"}, "not an image"),
        (ctx, {"image": "big.png"}, "image too large"),
        (ctx, {"image": "dir"}, "directory"),
        (one, {"image": "clipboard:2"}, "no such paste"),
        (ctx, {"image": "clipboard:1"}, "no such paste"),
        (ctx, {"references": [".env"]}, "refused: protected path"),
    ]
    for context, args, summary in cases:
        result = await _refused(tool, context, **args)
        assert result.summary == summary, (args, result)
    assert "clipboard:1" in (await _refused(tool, one, image="clipboard:2")).output


async def test_the_same_file_as_source_and_reference_is_an_error(tool: EditImageTool, ctx: ToolContext) -> None:
    result = await _refused(tool, ctx, references=["./src.png"])
    assert "src.png" in result.output and "twice" in result.output
    pasted = dataclasses.replace(ctx, pasted=(Image("png", png(8, 8), "clipboard"),))
    assert (await _refused(tool, pasted, image="clipboard:1", references=["clipboard:1"])).is_error


async def test_a_jpeg_and_a_webp_input_are_accepted(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "p.jpg").write_bytes(jpeg(64, 32))
    (workspace / "q.webp").write_bytes(webp_lossy(16, 16))
    run = await _prepared(tool, ctx, image="p.jpg", references=["q.webp"])
    assert run.preview is not None
    assert "source: p.jpg (JPEG, 64\u00d732, " in run.preview.body
    assert "reference: q.webp (WEBP, 16\u00d716, " in run.preview.body


async def test_the_preview_lists_prompt_inputs_size_model_and_flags(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "ref.png").write_bytes(png(32, 32))
    (workspace / "out.png").write_bytes(b"old")
    pasted = dataclasses.replace(ctx, pasted=(Image("png", png(8, 6), "clipboard"),))
    run = await _prepared(tool, pasted, references=["ref.png", "clipboard:1"], width=512, height=256, steps=5, seed=42, prompt="a\x1b[2Jb")
    assert run.target is None
    assert run.preview is not None and run.preview.kind == "content"
    assert run.preview.title == "edit image out.png"
    body = run.preview.body
    assert "\x1b" not in body
    assert "source: src.png (PNG, 64\u00d748, 1 KB)" in body
    assert "reference: ref.png (PNG, 32\u00d732, 1 KB)" in body
    assert "reference: clipboard:1 (PNG, 8\u00d76, 1 KB)" in body
    assert "size: 512\u00d7256, 5 steps, seed 42" in body
    assert "model: x/y" in body
    assert "[overwrites the existing file]" in body
    assert "model x/y is not downloaded; this call may download several GB" in body


async def test_no_download_line_when_the_model_is_cached(tool: EditImageTool, ctx: ToolContext, tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "w").write_text("x")
    run = await _prepared(tool, ctx)
    assert run.preview is not None and "not downloaded" not in run.preview.body
    assert "overwrites" not in run.preview.body


async def test_markup_is_drawn_literally(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "[bold]a[x].png").write_bytes(png(8, 8))
    run = await _prepared(tool, ctx, image="[bold]a[x].png", path="[bold]o[x].png", prompt="[bold]x[/]")
    assert run.preview is not None
    assert run.preview.title == "edit image [bold]o[x].png"
    assert "prompt: [bold]x[/]" in run.preview.body and "source: [bold]a[x].png" in run.preview.body
    result = await run.execute()
    assert result.output.startswith("edited [bold]o[x].png (")


async def test_the_edit_model_is_the_generators(make_tool: Any, ctx: ToolContext) -> None:
    run = await _prepared(make_tool(model="my/edit-model"), ctx)
    assert run.preview is not None and "model: my/edit-model" in run.preview.body


async def test_success_stages_the_checked_bytes_and_runs_the_edit_command(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.setenv("GITHUB_TOKEN", "sekrit-value-1234")
    ref = png(32, 32) + b"ref"
    (workspace / "ref.png").write_bytes(ref)
    run = await _prepared(tool, ctx, references=["ref.png"], width=256, height=256, steps=4, seed=7)
    (workspace / "src.png").write_bytes(b"changed after the checks")
    result = await run.execute()
    seen = json.loads(dump.read_text())
    argv = seen["argv"]
    out = argv[argv.index("--output") + 1]
    folder = Path(out).parent
    assert argv[:2] == ["--model", "x/y"] and argv[2] == "--prompt=make the sky red"
    assert argv[3] == "--image-paths" and argv[6] == "--width"
    assert argv[argv.index("--quantize") : argv.index("--quantize") + 2] == ["--quantize", "4"]
    assert argv[-1] == "--no-exif"
    first, second = seen["inputs"]
    assert Path(first).parent == folder and Path(second).parent == folder
    assert seen["shas"][first] == hashlib.sha256(png(64, 48)).hexdigest()  # the checked bytes, not the later file
    assert seen["shas"][second] == hashlib.sha256(ref).hexdigest()
    assert "GITHUB_TOKEN" not in seen["env"]
    assert not folder.exists()
    assert not result.is_error, result
    size = (workspace / "out.png").stat().st_size
    assert result.output == f"edited out.png (PNG, 256\u00d7256, {max(round(size / 1024), 1)} KB, seed 7)"


async def test_a_pasted_input_is_staged_with_exactly_the_pasted_bytes(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    data = png(40, 40) + b"pasted"
    pasted = dataclasses.replace(ctx, pasted=(Image("png", data, "clipboard"),))
    result = await (await _prepared(tool, pasted, image="clipboard:1")).execute()
    seen = json.loads(dump.read_text())
    assert not result.is_error, result
    assert seen["shas"][seen["inputs"][0]] == hashlib.sha256(data).hexdigest()


async def test_the_source_can_be_the_output_and_the_journal_is_told(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    journal = Journal()
    run = await _prepared(tool, dataclasses.replace(ctx, journal=journal), path="src.png")
    result = await run.execute()
    target = workspace / "src.png"
    assert not result.is_error and png_size(target.read_bytes()) == (256, 256)  # the source's size, raised to the minimum
    assert journal.calls == [("before", target), ("after", target)]


async def test_the_temp_folder_is_removed_after_an_error(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and list(temp.iterdir()) == []


async def test_bad_outputs_give_the_generate_image_errors_and_keep_the_target(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        ("wrongsize", "mflux wrote 128\u00d7128, expected 256\u00d7256"),
        ("jpeg", "mflux wrote a JPEG instead of a PNG"),
        ("nothing", "mflux wrote no image"),
        ("truncated", "mflux wrote an incomplete PNG"),
    ]
    (workspace / "out.png").write_bytes(b"old")
    for mode, message in cases:
        monkeypatch.setenv("FAKE_MFLUX_MODE", mode)
        result = await (await _prepared(tool, ctx, width=256, height=256)).execute()
        assert result.is_error and result.output == message, mode
        assert (workspace / "out.png").read_bytes() == b"old", mode


async def test_progress_lines_follow_the_phases(tool: EditImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.PROGRESS_TICK", 0.02)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    monkeypatch.setenv("FAKE_MFLUX_DELAY", "0.1")
    events: list[Any] = []
    run = await _prepared(tool, dataclasses.replace(ctx, progress=events.append), width=256, height=256, steps=4)
    result = await run.execute()
    assert not result.is_error, result
    texts = [e.text for e in events]
    assert all(isinstance(e, ToolProgress) for e in events)
    assert texts[0].startswith("starting") and any(t.startswith("step 2/4 ") for t in texts)


async def test_a_timeout_reads_timed_out_after_1_s(make_tool: Any, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(tmp_path / "pids"))
    result = await (await _prepared(make_tool(timeout=1), ctx)).execute()
    assert result.output == "timed out after 1 s" and result.summary == "timed out"


async def test_a_cancel_kills_the_process_group(tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine.test_imagegen_run import _gone

    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    task = asyncio.ensure_future((await _prepared(tool, ctx)).execute())
    for _ in range(250):
        if pids.exists() and len(pids.read_text().split()) == 2:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for pid in map(int, pids.read_text().split()):
        for _ in range(100):
            if _gone(pid):
                break
            await asyncio.sleep(0.02)
        assert _gone(pid)
    assert not (workspace / "out.png").exists()


def test_the_description_says_clipboard_labels_cover_only_the_latest_message() -> None:
    assert "latest user message" in EditImageTool.description


def test_the_label_names_the_output_not_the_prompt(tool: EditImageTool) -> None:
    assert tool.label({"prompt": "long " * 100, "image": "a.png", "path": "o/x.png"}) == "edit_image o/x.png"
    assert tool.label({"prompt": "p"}) == "edit_image"



async def test_a_cancel_during_staging_waits_for_the_thread_and_removes_the_temp_folder(
    tool: EditImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    from src.engine.tools import image as image_module

    folders: list[Path] = []
    started = threading.Event()
    real_stage = image_module.stage

    def slow_stage(inputs: Any, folder: Path) -> Any:
        folders.append(folder)
        started.set()
        time.sleep(0.3)
        (folder / "late").mkdir(parents=True, exist_ok=True)  # a write that lands late recreates the folder
        return real_stage(inputs, folder)

    monkeypatch.setattr(image_module, "stage", slow_stage)
    task = asyncio.ensure_future((await _prepared(tool, ctx)).execute())
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)  # a thread that was left running would have made the folder again by now
    assert folders and not folders[0].exists()
