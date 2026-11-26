"""upscale_image: validation, source checks, output size rule, dialog and execution against a fake mflux (phase A)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from src.engine.events import ToolProgress
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import UPSCALE_COMMAND, UPSCALE_EXTRA, MfluxGenerator, png_size
from src.engine.provider.base import Image
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.image import UpscaleImageTool

from tests.image_fakes import fake_mflux, png


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


def huge_png(width: int, height: int) -> bytes:
    """A PNG header that claims a size (the IDAT is not valid): only the dimensions are read."""
    data = bytearray(png(8, 8))
    data[16:24] = struct.pack(">II", width, height)
    return bytes(data)


@pytest.fixture
def make_tool(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    command = fake_mflux(folder, UPSCALE_COMMAND)

    def make(model: str = "x/y", timeout: float = 20) -> UpscaleImageTool:
        generator = MfluxGenerator(
            command, model, timeout, Guardrails.off(workspace), extra=UPSCALE_EXTRA, kind="upscale"
        )
        return UpscaleImageTool(generator)

    return make


@pytest.fixture
def tool(make_tool: Any) -> UpscaleImageTool:
    return make_tool()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    (workspace / "a.png").write_bytes(png(512, 512))
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))


async def _prepare(tool: UpscaleImageTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("image", "a.png")
    args.setdefault("path", "b.png")
    return await tool.prepare(args, ctx)


async def _refused(tool: UpscaleImageTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult), "a refusal must not produce a Prepared"
    assert result.is_error
    return result


async def _prepared(tool: UpscaleImageTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


def test_the_tool_asks_every_time_hides_in_plan_mode_and_can_be_interrupted(tool: UpscaleImageTool) -> None:
    assert tool.name == "upscale_image"
    assert tool.mutating and tool.needs_approval
    assert tool.runs_to_completion() is False
    assert tool.grant_arg == "path"
    props = tool.parameters["properties"]
    assert props["scale"]["enum"] == [2, 3, 4]
    assert tool.parameters["required"] == ["image", "path"]


def test_the_label_names_the_output(tool: UpscaleImageTool) -> None:
    assert tool.label({"image": "a.png", "path": "o/x.png"}) == "upscale_image o/x.png"
    assert tool.label({}) == "upscale_image"


async def test_success_runs_the_upscaler_on_the_staged_source(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.setenv("GITHUB_TOKEN", "sekrit-value-1234")
    run = await _prepared(tool, ctx, seed=7)
    (workspace / "a.png").write_bytes(b"changed after the checks")
    result = await run.execute()
    seen = json.loads(dump.read_text())
    argv = seen["argv"]

    def after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    folder = Path(after("--output")).parent
    assert argv[:2] == ["--model", "x/y"]
    assert after("--resolution") == "2x" and after("--seed") == "7" and after("--softness") == "0"
    assert argv[argv.index("--low-ram") : argv.index("--low-ram") + 3] == ["--low-ram", "--quantize", "8"]
    assert argv[-1] == "--no-exif" and "--prompt" not in " ".join(argv)
    (staged,) = seen["inputs"]
    assert Path(staged).parent == folder and after("--image-path") == staged
    assert seen["shas"][staged] == hashlib.sha256(png(512, 512)).hexdigest()
    assert "GITHUB_TOKEN" not in seen["env"]
    assert not folder.exists()
    assert not result.is_error, result
    size = (workspace / "b.png").stat().st_size
    assert result.output == f"upscaled b.png (PNG, 1024\u00d71024, {max(round(size / 1024), 1)} KB, seed 7)"


async def test_the_scale_and_softness_reach_the_command(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    (workspace / "s.png").write_bytes(png(256, 256))
    result = await (await _prepared(tool, ctx, image="s.png", scale=4, softness=0.25)).execute()
    argv = json.loads(dump.read_text())["argv"]
    assert argv[argv.index("--resolution") + 1] == "4x" and argv[argv.index("--softness") + 1] == "0.25"
    assert png_size((workspace / "b.png").read_bytes()) == (1024, 1024), result


async def test_bad_arguments_are_invalid_errors_naming_the_argument(tool: UpscaleImageTool, ctx: ToolContext) -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"scale": 5}, "scale"),
        ({"scale": 1}, "scale"),
        ({"scale": "2"}, "scale"),
        ({"softness": 1.5}, "softness"),
        ({"softness": -0.1}, "softness"),
        ({"softness": "soft"}, "softness"),
        ({"softness": True}, "softness"),
        ({"softness": float("nan")}, "softness"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"path": "a.jpg"}, "a.jpg"),
        ({"image": None}, "image"),
    ]
    for args, name in cases:
        result = await _refused(tool, ctx, **args)
        assert result.summary == "invalid" and name in result.output, (args, result)


async def test_an_output_over_the_limit_is_refused_before_any_process(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    (workspace / "big.png").write_bytes(png(1100, 1100))
    result = await _refused(tool, ctx, image="big.png")
    assert result.summary == "invalid"
    assert result.output == (
        "the output would be 2192\u00d72192; the longer side may be at most 2048 (use a smaller scale)"
    )
    (workspace / "wide.png").write_bytes(png(1500, 300))
    assert "the output would be 2960\u00d7592" in (await _refused(tool, ctx, image="wide.png")).output
    assert not dump.exists()


async def test_the_limit_itself_is_allowed(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "edge.png").write_bytes(png(1024, 1024))
    run = await _prepared(tool, ctx, image="edge.png")
    assert run.preview is not None and "output: 2048\u00d72048 (2\u00d7)" in run.preview.body


async def test_a_source_over_50_megapixels_is_refused_before_staging(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "huge.png").write_bytes(huge_png(8000, 7000))
    result = await _refused(tool, ctx, image="huge.png")
    assert "50 megapixels" in result.output and result.summary == "invalid"


async def test_refused_sources_never_reach_a_dialog(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
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
    ]
    for context, args, summary in cases:
        result = await _refused(tool, context, **args)
        assert result.summary == summary, (args, result)
    assert "clipboard:1" in (await _refused(tool, one, image="clipboard:2")).output


async def test_a_pasted_source_is_staged_with_exactly_the_pasted_bytes(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    data = png(40, 40) + b"pasted"
    pasted = dataclasses.replace(ctx, pasted=(Image("png", data, "clipboard"),))
    result = await (await _prepared(tool, pasted, image="clipboard:1")).execute()
    seen = json.loads(dump.read_text())
    assert not result.is_error, result
    assert seen["shas"][seen["inputs"][0]] == hashlib.sha256(data).hexdigest()
    assert png_size((workspace / "b.png").read_bytes()) == (80, 80)


async def test_the_preview_lists_source_output_softness_seed_model_and_flags(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "b.png").write_bytes(b"old")
    run = await _prepared(tool, ctx, scale=3, softness=0.5, seed=42)
    assert run.target is None
    assert run.preview is not None and run.preview.kind == "content"
    assert run.preview.title == "upscale image b.png"
    body = run.preview.body
    assert "source: a.png (PNG, 512\u00d7512, 2 KB)" in body
    assert "output: 1536\u00d71536 (3\u00d7)" in body
    assert "softness: 0.5" in body
    assert "seed: 42" in body
    assert "model: x/y" in body
    assert "[overwrites the existing file]" in body
    assert "model x/y is not downloaded; this call may download several GB" in body


async def test_softness_is_left_out_of_the_preview_when_zero(tool: UpscaleImageTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx)
    assert run.preview is not None
    assert "softness" not in run.preview.body and "overwrites" not in run.preview.body


@pytest.mark.parametrize("scale, short", [(2, n) for n in range(1, 8)] + [(3, n) for n in range(1, 6)] + [(4, n) for n in range(1, 4)])
async def test_a_source_too_small_for_the_scale_is_refused_before_any_process(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, scale: int, short: int
) -> None:
    (workspace / "tiny.png").write_bytes(png(short, 300))
    result = await _refused(tool, ctx, image="tiny.png", scale=scale)
    assert result.summary == "invalid"
    assert "the output would be 0\u00d70" in result.output
    assert f"{scale}\u00d7" in result.output and "the shorter side must be at least" in result.output


@pytest.mark.parametrize("scale, short", [(2, 8), (3, 6), (4, 4)])
async def test_the_smallest_workable_source_is_accepted(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, scale: int, short: int
) -> None:
    (workspace / "tiny.png").write_bytes(png(short, short))
    await _prepared(tool, ctx, image="tiny.png", scale=scale)


async def test_the_dialog_uses_the_oriented_size_rule_for_the_output(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "odd.png").write_bytes(png(333, 250))
    run = await _prepared(tool, ctx, image="odd.png")
    assert run.preview is not None and "output: 660\u00d7496 (2\u00d7)" in run.preview.body
    result = await run.execute()
    assert png_size((workspace / "b.png").read_bytes()) == (660, 496), result


async def test_the_download_line_follows_the_known_model_files(
    make_tool: Any, ctx: ToolContext, tmp_path: Path
) -> None:
    repo = tmp_path / "hub" / "models--numz--SeedVR2_comfyUI" / "snapshots" / "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"
    repo.mkdir(parents=True)
    (repo / "ema_vae_fp16.safetensors").write_text("x")
    run = await _prepared(make_tool(model="seedvr2-3b"), ctx)
    assert run.preview is not None and "model seedvr2-3b is not downloaded" in run.preview.body
    (repo / "seedvr2_ema_3b_fp16.safetensors").write_text("x")
    run = await _prepared(make_tool(model="seedvr2-3b"), ctx)
    assert run.preview is not None and "not downloaded" not in run.preview.body


async def test_the_source_can_be_the_output_and_the_journal_is_told(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path) -> None:
    journal = Journal()
    run = await _prepared(tool, dataclasses.replace(ctx, journal=journal), path="a.png")
    result = await run.execute()
    target = workspace / "a.png"
    assert not result.is_error and png_size(target.read_bytes()) == (1024, 1024)
    assert journal.calls == [("before", target), ("after", target)]


async def test_a_new_output_is_journalled_so_undo_can_remove_it(tool: UpscaleImageTool, ctx: ToolContext, workspace: Path) -> None:
    journal = Journal()
    result = await (await _prepared(tool, dataclasses.replace(ctx, journal=journal))).execute()
    assert not result.is_error
    assert journal.calls == [("before", workspace / "b.png"), ("after", workspace / "b.png")]


async def test_the_temp_folder_is_removed_after_an_error(
    tool: UpscaleImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    result = await (await _prepared(tool, ctx)).execute()
    assert result.is_error and result.output.startswith("mflux failed (exit 1)") and list(temp.iterdir()) == []


async def test_bad_outputs_give_the_shared_errors_and_keep_the_target(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        ("wrongsize", "mflux wrote 512\u00d7512, expected 1024\u00d71024"),
        ("jpeg", "mflux wrote a JPEG instead of a PNG"),
        ("nothing", "mflux wrote no image"),
        ("truncated", "mflux wrote an incomplete PNG"),
    ]
    (workspace / "b.png").write_bytes(b"old")
    for mode, message in cases:
        monkeypatch.setenv("FAKE_MFLUX_MODE", mode)
        result = await (await _prepared(tool, ctx)).execute()
        assert result.is_error and result.output == message, mode
        assert (workspace / "b.png").read_bytes() == b"old", mode


async def test_progress_lines_follow_a_one_step_run(tool: UpscaleImageTool, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.PROGRESS_TICK", 0.02)
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    monkeypatch.setenv("FAKE_MFLUX_DELAY", "0.1")
    events: list[Any] = []
    result = await (await _prepared(tool, dataclasses.replace(ctx, progress=events.append))).execute()
    assert not result.is_error, result
    texts = [e.text for e in events]
    assert all(isinstance(e, ToolProgress) for e in events)
    assert texts[0].startswith("starting") and any(t.startswith(("step 1/1 ", "saving")) for t in texts)  # a one-step run reaches saving at once


async def test_a_timeout_reads_timed_out_after_1_s(make_tool: Any, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(tmp_path / "pids"))
    result = await (await _prepared(make_tool(timeout=1), ctx)).execute()
    assert result.output == "timed out after 1 s" and result.summary == "timed out"


async def test_a_cancel_kills_the_process_group(
    tool: UpscaleImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.engine.test_imagegen_run import _gone

    pids = tmp_path / "pids"
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
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
    assert not (workspace / "b.png").exists() and list(temp.iterdir()) == []


async def test_a_running_warm_up_refuses_the_call(make_tool: Any, ctx: ToolContext) -> None:
    from src.engine.image_pull import REFUSED

    class Busy:
        busy = True

    tool = make_tool()
    tool.pull = Busy()
    result = await _refused(tool, ctx)
    assert result.output == REFUSED and result.summary == "refused"
