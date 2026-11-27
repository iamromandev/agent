"""edit_image(upscale=true): refusals, dialog and the extra SeedVR2 step between the edit and the blend (phase B)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from core.engine.guardrails.setup import Guardrails
from core.engine.imagegen import EDIT_COMMAND, EDIT_EXTRA, UPSCALE_COMMAND, UPSCALE_EXTRA, MfluxGenerator, png_size
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.image import EditImageTool
from core.testing.image_fakes import fake_composite, fake_mflux, png

REGION = [{"shape": "rect", "x": 0, "y": 0, "w": 100, "h": 100}]


def _huge(width: int, height: int) -> bytes:
    data = bytearray(png(8, 8))
    data[16:24] = width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return bytes(data)


@pytest.fixture
def env(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(tmp_path / "edit.json"))
    monkeypatch.setenv("FAKE_UPSCALE_DUMP", str(tmp_path / "up.json"))
    helper = fake_composite(folder)
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", helper)
    return {
        "edit": fake_mflux(folder, EDIT_COMMAND),
        "up": fake_mflux(folder, UPSCALE_COMMAND, "FAKE_UPSCALE"),
        "helper": helper,
    }


def _tool(
    env: dict[str, Path], workspace: Path, timeout: float = 30, upscaler: bool = True, up_model: str = "seedvr2-3b"
) -> EditImageTool:
    guard = Guardrails.off(workspace)
    edit = MfluxGenerator(env["edit"], "x/y", timeout, guard, extra=EDIT_EXTRA)
    up = MfluxGenerator(env["up"], up_model, timeout, guard, extra=UPSCALE_EXTRA, kind="upscale") if upscaler else None
    return EditImageTool(edit, None, up)


@pytest.fixture
def tool(env: dict[str, Path], workspace: Path) -> EditImageTool:
    return _tool(env, workspace)


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    (workspace / "big.png").write_bytes(_huge(3000, 2000))
    (workspace / "small.png").write_bytes(png(64, 48))
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))


async def _prepare(tool: EditImageTool, ctx: ToolContext, **args: Any) -> Prepared | ToolResult:
    args.setdefault("prompt", "sharpen")
    args.setdefault("image", "big.png")
    args.setdefault("path", "out.png")
    return await tool.prepare(args, ctx)


async def _refused(tool: EditImageTool, ctx: ToolContext, **args: Any) -> ToolResult:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, ToolResult)
    assert result.is_error and result.summary == "invalid"
    return result


async def _prepared(tool: EditImageTool, ctx: ToolContext, **args: Any) -> Prepared:
    result = await _prepare(tool, ctx, **args)
    assert isinstance(result, Prepared), result
    return result


async def _go(run: Prepared) -> ToolResult:
    return await run.execute()


def _dump(tmp_path: Path, name: str) -> dict[str, Any]:
    return json.loads((tmp_path / name).read_text())


def _opt(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


async def test_upscale_without_region_or_mask_is_refused(tool: EditImageTool, ctx: ToolContext, tmp_path: Path) -> None:
    result = await _refused(tool, ctx, upscale=True)
    assert result.output.startswith("upscale needs a region or mask")
    assert not (tmp_path / "edit.json").exists()


async def test_upscale_with_a_source_no_larger_than_the_edit_is_refused(tool: EditImageTool, ctx: ToolContext) -> None:
    result = await _refused(tool, ctx, image="small.png", region=REGION, upscale=True)
    assert "upscale is only useful when the source is larger than the edit (256\u00d7256)" in result.output


async def test_upscale_without_an_upscaler_is_refused(env: dict[str, Path], workspace: Path, ctx: ToolContext) -> None:
    result = await _refused(_tool(env, workspace, upscaler=False), ctx, region=REGION, upscale=True)
    assert result.output.startswith("upscale model not available")


async def test_upscale_false_is_the_same_as_absent(tool: EditImageTool, ctx: ToolContext) -> None:
    plain = await _prepared(tool, ctx, region=REGION)
    off = await _prepared(tool, ctx, region=REGION, upscale=False, seed=1)
    assert plain.preview is not None and off.preview is not None
    assert "upscale" not in off.preview.body
    assert [line for line in plain.preview.body.splitlines() if not line.startswith("size:")] == [
        line for line in off.preview.body.splitlines() if not line.startswith("size:")
    ]


async def test_the_dialog_names_the_factor_and_the_upscale_models_download_state(
    env: dict[str, Path], workspace: Path, ctx: ToolContext
) -> None:
    run = await _prepared(_tool(env, workspace, up_model="a/b\x1b"), ctx, region=REGION, upscale=True)
    assert run.preview is not None
    lines = run.preview.body.splitlines()
    assert "upscale: 2\u00d7 (a/b\\x1b)" in lines
    assert lines.index("upscale: 2\u00d7 (a/b\\x1b)") > next(i for i, x in enumerate(lines) if x.startswith("output:"))
    assert "[model x/y is not downloaded; this call may download several GB]" in lines
    assert "[model a/b\\x1b is not downloaded; this call may download several GB]" in lines


async def test_an_upscaled_edit_runs_the_upscaler_on_the_staged_edit_with_the_same_seed(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    result = await (await _prepared(tool, ctx, region=REGION, upscale=True, seed=77)).execute()
    assert not result.is_error, result.output
    assert result.output.endswith("seed 77, masked, upscaled 2\u00d7)")
    edit, up = _dump(tmp_path, "edit.json"), _dump(tmp_path, "up.json")
    assert (_opt(edit["argv"], "--width"), _opt(edit["argv"], "--height")) == ("1024", "688")
    assert _opt(up["argv"], "--resolution") == "2x"
    assert _opt(up["argv"], "--seed") == "77" == _opt(edit["argv"], "--seed")
    assert Path(_opt(up["argv"], "--image-path")).name == "edit.png"
    assert png_size((workspace / "out.png").read_bytes()) == (3000, 2000)


async def test_the_edit_passed_to_the_blend_is_the_upscaled_image(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_COMPOSITE_DUMP", str(tmp_path / "comp.jsonl"))
    await (await _prepared(tool, ctx, region=REGION, upscale=True)).execute()
    runs = [json.loads(x) for x in (tmp_path / "comp.jsonl").read_text().splitlines()]
    blend = next(r for r in runs if r["step"] == "blend")
    assert Path(_opt(blend["argv"], "--edit")).name == "image.png"  # the upscaler's output, not the staged edit.png


async def test_a_timeout_in_the_upscale_step_names_the_whole_timeout(
    env: dict[str, Path], workspace: Path, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_UPSCALE_MODE", "sleep")
    monkeypatch.setenv("FAKE_UPSCALE_PIDS", str(tmp_path / "pids"))
    result = await (await _prepared(_tool(env, workspace, timeout=3), ctx, region=REGION, upscale=True)).execute()
    assert result.is_error and result.output == "timed out after 3 s"
    assert not (workspace / "out.png").exists()


async def test_a_wrong_sized_upscale_output_keeps_the_target_and_removes_the_folder(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv("FAKE_UPSCALE_MODE", "wrongsize")
    (workspace / "out.png").write_bytes(b"old")
    result = await (await _prepared(tool, ctx, region=REGION, upscale=True)).execute()
    assert result.is_error and result.output == "mflux wrote 1024\u00d7688, expected 2048\u00d71376"
    assert (workspace / "out.png").read_bytes() == b"old"
    assert list(temp.iterdir()) == []


async def test_esc_during_the_upscale_step_kills_it_and_removes_the_folder(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_UPSCALE_MODE", "sleep")
    monkeypatch.setenv("FAKE_UPSCALE_PIDS", str(pids))
    task = asyncio.create_task(_go(await _prepared(tool, ctx, region=REGION, upscale=True)))
    for _ in range(200):
        if pids.exists() and pids.read_text():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pids.read_text().split()[0])
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert list(temp.iterdir()) == []


async def test_the_real_helper_keeps_every_pixel_outside_the_region(
    env: dict[str, Path], workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pil = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(
        "core.engine.tools.image.COMPOSITE_HELPER",
        Path(__file__).resolve().parents[3] / "core" / "engine" / "composite_helper.py",
    )
    monkeypatch.setenv("FAKE_UPSCALE_COLOR", "255,0,0")
    x = np.arange(3000, dtype=np.uint16)[None, :].repeat(2000, 0).astype(np.uint8)
    y = np.arange(2000, dtype=np.uint16)[:, None].repeat(3000, 1).astype(np.uint8)
    base = np.stack([x, y, x ^ y], axis=2)
    pil.fromarray(base, "RGB").save(workspace / "photo.png")
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))
    region = [{"shape": "rect", "x": 0, "y": 0, "w": 1000, "h": 2000}]
    run = await _prepared(_tool(env, workspace), ctx, image="photo.png", region=region, feather=0, upscale=True)
    result = await run.execute()
    assert not result.is_error, result.output
    with pil.open(workspace / "out.png") as image:
        out = np.asarray(image.convert("RGB"))
    assert out.shape == (2000, 3000, 3)
    assert (out[:, :1000] == (255, 0, 0)).all()
    assert (out[:, 1000:] == base[:, 1000:]).all()


async def test_a_source_barely_larger_than_the_rounded_edit_is_refused(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "wide.png").write_bytes(_huge(1000, 500))
    result = await _refused(tool, ctx, image="wide.png", region=REGION, upscale=True)
    assert "upscale is only useful when the source is larger than the edit (1008\u00d7496)" in result.output


async def test_a_larger_source_is_accepted(tool: EditImageTool, ctx: ToolContext) -> None:
    await _prepared(tool, ctx, region=REGION, upscale=True)


@pytest.mark.parametrize(("mode", "text"), [("fail", "mflux failed (exit 1)"), ("nothing", "mflux wrote no image")])
async def test_an_upscaler_failure_is_reported_and_keeps_the_target(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch, mode: str, text: str
) -> None:
    monkeypatch.setenv("FAKE_UPSCALE_MODE", mode)
    (workspace / "out.png").write_bytes(b"old")
    result = await (await _prepared(tool, ctx, region=REGION, upscale=True)).execute()
    assert result.is_error and result.output.startswith(text), result.output
    assert (workspace / "out.png").read_bytes() == b"old"


async def test_a_deadline_gone_before_the_upscale_step_never_starts_the_upscaler(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock:
        @staticmethod
        def monotonic() -> float:
            return 1000.0 if (tmp_path / "edit.json").exists() else 0.0

    monkeypatch.setattr("core.engine.tools.image.time", Clock)
    result = await (await _prepared(tool, ctx, region=REGION, upscale=True)).execute()
    assert result.is_error and result.output == "timed out after 30 s" and result.summary == "timed out"
    assert not (tmp_path / "up.json").exists()


async def test_a_mask_image_can_be_upscaled(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "mask.png").write_bytes(png(32, 24))
    result = await (await _prepared(tool, ctx, mask="mask.png", upscale=True)).execute()
    assert not result.is_error, result.output
    assert result.output.endswith("masked, upscaled 2\u00d7)")


async def test_an_edit_that_cannot_be_upscaled_is_refused_before_any_process(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("core.engine.tools.image.edit_upscale_scale", lambda *a, **k: None)
    result = await _refused(tool, ctx, region=REGION, upscale=True)
    assert "upscale is not possible for an edit of 1024\u00d7688" in result.output
    assert not (tmp_path / "edit.json").exists()


async def test_a_running_warm_up_refuses_an_upscaled_edit(tool: EditImageTool, ctx: ToolContext) -> None:
    from core.engine.image_pull import REFUSED

    class Busy:
        busy = True

    setattr(tool, "pull", Busy())  # noqa: B010
    result = await _prepare(tool, ctx, region=REGION, upscale=True)
    assert isinstance(result, ToolResult) and result.output == REFUSED and result.summary == "refused"


async def test_a_non_boolean_upscale_is_invalid(tool: EditImageTool, ctx: ToolContext) -> None:
    await _refused(tool, ctx, region=REGION, upscale="yes")


async def test_a_masked_edit_without_upscale_never_starts_the_upscaler(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path
) -> None:
    result = await (await _prepared(tool, ctx, region=REGION)).execute()
    assert not result.is_error and result.output.endswith(", masked)")
    assert not (tmp_path / "up.json").exists()


async def test_without_a_deadline_the_upscale_step_has_none_either(
    env: dict[str, Path], workspace: Path, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float | None] = []
    from core.engine.tools import image as module

    real = module._generate

    async def spy(generator: Any, *a: Any, **k: Any) -> Any:
        seen.append(generator.timeout)
        return await real(generator, *a, **k)

    monkeypatch.setattr(module, "_generate", spy)
    tool = _tool(env, workspace)
    tool.generator.timeout = None
    assert tool.upscaler is not None
    tool.upscaler.timeout = 5
    result = await (await _prepared(tool, ctx, region=REGION, upscale=True)).execute()
    assert not result.is_error, result.output
    assert seen == [None, None]
