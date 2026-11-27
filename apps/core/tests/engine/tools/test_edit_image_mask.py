"""edit_image with a region or mask: validation, dialog, helper wiring, failures, cancel (AC1, AC5-AC13, AC15)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from core.engine.guardrails.setup import Guardrails
from core.engine.imagegen import EDIT_COMMAND, EDIT_EXTRA, MfluxGenerator, png_size
from core.engine.provider.base import Image
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.image import EditImageTool
from core.testing.image_fakes import fake_composite, fake_mflux, png

from tests.engine.test_image_orientation import exif_jpeg

REGION = [{"shape": "rect", "x": 0, "y": 0, "w": 32, "h": 48}]


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


@pytest.fixture
def env(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(tmp_path / "mflux.json"))
    monkeypatch.setenv("FAKE_COMPOSITE_DUMP", str(tmp_path / "comp.jsonl"))
    helper = fake_composite(folder)
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", helper)
    return {"mflux": fake_mflux(folder, EDIT_COMMAND), "helper": helper}


def _tool(env: dict[str, Path], workspace: Path, guard: Guardrails | None = None, timeout: float = 30) -> EditImageTool:
    guard = guard or Guardrails.off(workspace)
    return EditImageTool(MfluxGenerator(env["mflux"], "x/y", timeout, guard, extra=EDIT_EXTRA))


@pytest.fixture
def tool(env: dict[str, Path], workspace: Path) -> EditImageTool:
    return _tool(env, workspace)


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    (workspace / "src.png").write_bytes(png(64, 48))
    (workspace / "mask.png").write_bytes(png(32, 24))
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


def _body(run: Prepared) -> str:
    assert run.preview is not None
    return run.preview.body


async def _go(run: Prepared) -> ToolResult:
    return await run.execute()


def _runs(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "comp.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"region": REGION, "mask": "mask.png"}, "either"),
        ({"feather": 4}, "feather"),
        ({"region": REGION, "feather": 65}, "feather"),
        ({"region": REGION, "feather": -1}, "feather"),
        ({"region": REGION * 9}, "region"),
        ({"region": [{**REGION[0], "w": 0}]}, "region shape 1"),
        ({"region": [{**REGION[0], "shape": "star"}]}, "region shape 1"),
        ({"region": []}, "region"),
        ({"region": REGION, "width": 512}, "width and height cannot be combined with a mask or region (the output keeps the source size)"),
        ({"mask": "mask.png", "height": 512}, "width and height cannot be combined"),
    ],
)
async def test_bad_selection_arguments_are_invalid_and_start_nothing(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, args: dict[str, Any], fragment: str
) -> None:
    result = await _refused(tool, ctx, **args)
    assert result.summary == "invalid"
    assert fragment in result.output
    assert not (tmp_path / "comp.jsonl").exists() and not (tmp_path / "mflux.json").exists()


async def test_a_shape_wholly_outside_the_source_is_refused_in_prepare(tool: EditImageTool, ctx: ToolContext) -> None:
    result = await _refused(tool, ctx, region=[REGION[0], {**REGION[0], "x": 64}])
    assert "region shape 2 lies outside the source (64\u00d748)" in result.output


async def test_the_oriented_size_of_a_jpeg_bounds_the_region(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "p.jpg").write_bytes(exif_jpeg(60, 40, 6))  # shown as 40x60
    await _prepared(tool, ctx, image="p.jpg", region=[{"shape": "rect", "x": 0, "y": 50, "w": 10, "h": 5}])
    result = await _refused(tool, ctx, image="p.jpg", region=[{"shape": "rect", "x": 45, "y": 0, "w": 5, "h": 5}])
    assert "(40\u00d760)" in result.output


async def test_the_dialog_names_the_region_and_the_source_size_output(tool: EditImageTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx, region=[*REGION, {**REGION[0], "shape": "ellipse"}], feather=3)
    body = _body(run)
    assert "region: 1 rect, 1 ellipse, feather 3 px" in body
    assert "output: 64\u00d748 (source size)" in body
    assert "size: 256\u00d7256, 4 steps" in body  # the edit itself runs at the default size


async def test_the_dialog_names_the_mask_with_the_default_feather(tool: EditImageTool, ctx: ToolContext) -> None:
    run = await _prepared(tool, ctx, mask="mask.png")
    assert "mask: mask.png, feather 8 px" in _body(run)
    assert "reference:" not in _body(run)


async def test_an_unmasked_call_has_no_selection_lines(tool: EditImageTool, ctx: ToolContext) -> None:
    body = _body(await _prepared(tool, ctx))
    assert "region" not in body and "mask" not in body and "output:" not in body


async def test_mask_refusals_use_the_source_reasons(tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(8, 8))
    (workspace / "link.png").symlink_to(outside)
    (workspace / ".env").write_text("A=1\n")
    (workspace / "text.png").write_text("hello")
    (workspace / "big.png").write_bytes(png(8, 8) + b"\0" * (6 * 1024 * 1024))
    cases = {
        str(outside): "refused: outside workspace",
        "link.png": "refused: outside workspace",
        ".env": "refused: protected path",
        "text.png": "not an image",
        "big.png": "image too large",
        "nope.png": "not found",
    }
    for ref, summary in cases.items():
        result = await _refused(tool, ctx, mask=ref)
        assert result.summary == summary, (ref, result)


async def test_the_mask_cannot_be_the_source_or_a_reference_or_given_twice(tool: EditImageTool, ctx: ToolContext) -> None:
    assert "given twice" in (await _refused(tool, ctx, mask="src.png")).output
    assert "given twice" in (await _refused(tool, ctx, references=["mask.png"], mask="mask.png")).output


async def test_the_mask_counts_toward_the_four_input_cap(tool: EditImageTool, ctx: ToolContext, workspace: Path) -> None:
    for name in ("a", "b", "c"):
        (workspace / f"{name}.png").write_bytes(png(8, 8 + ord(name)))
    result = await _refused(tool, ctx, references=["a.png", "b.png", "c.png"], mask="mask.png")
    assert "too many input images (at most 4 with the mask)" in result.output
    await _prepared(tool, ctx, references=["a.png", "b.png"], mask="mask.png")


async def test_a_pasted_mask_uses_the_numbered_paste(tool: EditImageTool, ctx: ToolContext) -> None:
    pasted = (Image("png", png(8, 8), "a"), Image("png", png(16, 12), "b"))
    run = await _prepared(tool, dataclasses.replace(ctx, pasted=pasted), mask="clipboard:2")
    assert "mask: clipboard:2" in _body(run)
    result = await _refused(tool, dataclasses.replace(ctx, pasted=pasted), mask="clipboard:3")
    assert "clipboard:1, clipboard:2" in result.output


async def test_a_masked_edit_runs_prepare_mflux_and_blend_and_says_masked(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    journal = Journal()
    (workspace / "out.png").write_bytes(b"old")
    run = await _prepared(tool, dataclasses.replace(ctx, journal=journal), region=REGION, feather=2, seed=5)
    result = await run.execute()
    assert not result.is_error, result.output
    assert result.output == f"edited out.png (PNG, 64\u00d748, {result.output.split(', ')[2]}, seed 5, masked)"
    assert png_size((workspace / "out.png").read_bytes()) == (64, 48)
    assert journal.calls == [("before", workspace / "out.png"), ("after", workspace / "out.png")]
    steps = _runs(tmp_path)
    assert [s["step"] for s in steps] == ["prepare", "blend"]
    mflux = json.loads((tmp_path / "mflux.json").read_text())
    assert mflux["argv"][mflux["argv"].index("--width") + 1] == "256"
    assert len(mflux["inputs"]) == 1  # the mask never goes to mflux


async def test_an_unmasked_call_starts_no_helper_and_keeps_the_result(tool: EditImageTool, ctx: ToolContext, tmp_path: Path) -> None:
    result = await (await _prepared(tool, ctx, seed=5)).execute()
    assert result.output.endswith("seed 5)") and "masked" not in result.output
    assert not (tmp_path / "comp.jsonl").exists()


async def test_a_large_source_runs_the_edit_at_the_default_size_and_returns_the_source_size(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "big.png").write_bytes(png(4000, 3000))
    result = await (await _prepared(tool, ctx, image="big.png", region=[{**REGION[0], "w": 2000, "h": 3000}])).execute()
    assert not result.is_error, result.output
    mflux = json.loads((tmp_path / "mflux.json").read_text())
    argv = mflux["argv"]
    assert (argv[argv.index("--width") + 1], argv[argv.index("--height") + 1]) == ("1024", "768")
    assert png_size((workspace / "out.png").read_bytes()) == (4000, 3000)


async def test_a_mask_is_staged_as_its_own_copy_and_passed_by_private_path(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    result = await (await _prepared(tool, ctx, mask="mask.png")).execute()
    assert not result.is_error, result.output
    prepare = _runs(tmp_path)[0]
    folder = Path(prepare["cwd"]).resolve()
    argv = prepare["argv"]
    for flag in ("--source", "--mask", "--out-mask"):
        assert Path(argv[argv.index(flag) + 1]).resolve().parent == folder
    assert argv[argv.index("--expect") + 1 : argv.index("--expect") + 3] == ["64", "48"]
    assert argv[argv.index("--feather") + 1] == "8"


async def test_the_helper_runs_isolated_without_secrets(
    env: dict[str, Path], ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = Guardrails.off(workspace)
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecretsecret")
    result = await (await _prepared(_tool(env, workspace, guard), ctx, region=REGION)).execute()
    assert not result.is_error, result.output
    steps = _runs(tmp_path)
    assert len(steps) == 2
    for step in steps:
        assert step["flags"] == 1 and "HF_TOKEN" not in step["env"]
        assert os.path.samefile(step["exe"], sys.executable) or step["exe"].startswith(sys.prefix)


@pytest.mark.parametrize(
    ("mode", "step", "message"),
    [
        ("fail", "blend", "compositing failed (exit 1): boom"),
        ("fail", "prepare", "compositing failed (exit 1): boom"),
        ("wrongsize", "blend", "compositing wrote 65\u00d748, expected 64\u00d748"),
        ("nofile", "blend", "compositing wrote no image"),
        ("text", "blend", "compositing wrote an unknown file instead of a PNG"),
        ("nothing", "prepare", "the mask selects nothing"),
        ("orientation", "prepare", "the source decoded as 20\u00d710, expected 10\u00d720"),
        ("toolarge", "prepare", "image too large to composite (limit 50 megapixels)"),
    ],
)
async def test_helper_failures_are_errors_that_keep_the_target(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, step: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_COMPOSITE_MODE", mode)
    monkeypatch.setenv("FAKE_COMPOSITE_STEP", step)
    (workspace / "out.png").write_bytes(b"old")
    result = await (await _prepared(tool, ctx, region=REGION)).execute()
    assert result.is_error and message in result.output, result.output
    assert (workspace / "out.png").read_bytes() == b"old"
    if step == "prepare":
        assert not (tmp_path / "mflux.json").exists(), "mflux must not start"


async def test_the_temp_folder_is_gone_after_success_failure_and_cancel(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    await (await _prepared(tool, ctx, region=REGION)).execute()
    assert list(temp.iterdir()) == []
    monkeypatch.setenv("FAKE_COMPOSITE_MODE", "fail")
    await (await _prepared(tool, ctx, region=REGION)).execute()
    assert list(temp.iterdir()) == []


async def test_esc_during_compositing_kills_the_helper_and_removes_the_folder(
    tool: EditImageTool, ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    pids = tmp_path / "pid"
    monkeypatch.setenv("FAKE_COMPOSITE_MODE", "sleep")
    monkeypatch.setenv("FAKE_COMPOSITE_PIDS", str(pids))
    task = asyncio.create_task(_go(await _prepared(tool, ctx, region=REGION)))
    for _ in range(200):
        if pids.exists() and pids.read_text():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pids.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert list(temp.iterdir()) == []


async def test_one_deadline_covers_the_helper_too(
    env: dict[str, Path], ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_COMPOSITE_MODE", "sleep")
    monkeypatch.setenv("FAKE_COMPOSITE_PIDS", str(tmp_path / "pid"))
    result = await (await _prepared(_tool(env, workspace, timeout=2), ctx, region=REGION)).execute()
    assert result.is_error and result.output.startswith("timed out after")


def test_the_description_explains_region_mask_feather_and_the_source_size_output() -> None:
    text = EditImageTool.description.lower()
    props = EditImageTool.parameters["properties"]
    for word in ("region", "mask", "feather", "source"):
        assert word in text
    assert "source pixel" in props["region"]["description"].lower()
    assert "white" in props["mask"]["description"].lower()
    assert props["feather"]["type"] == "integer"
    assert props["region"]["maxItems"] == 8


# the real helper, end to end (needs Pillow)


async def test_the_real_helper_edits_the_left_half_and_keeps_the_right_exactly(
    env: dict[str, Path], workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pil = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", Path(__file__).resolve().parents[3] / "core" / "engine" / "composite_helper.py")
    monkeypatch.setenv("FAKE_MFLUX_COLOR", "255,0,0")
    x = np.arange(640, dtype=np.uint8)[None, :].repeat(480, 0)
    y = np.arange(480, dtype=np.uint8)[:, None].repeat(640, 1)
    base = np.stack([x, y, x ^ y], axis=2)
    pil.fromarray(base, "RGB").save(workspace / "photo.png")
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))
    region = [{"shape": "rect", "x": 0, "y": 0, "w": 320, "h": 480}]
    result = await (await _prepared(_tool(env, workspace), ctx, image="photo.png", region=region, feather=0)).execute()
    assert not result.is_error, result.output
    assert result.output.endswith(", masked)")
    with pil.open(workspace / "out.png") as image:
        out = np.asarray(image.convert("RGB"))
    assert out.shape == (480, 640, 3)
    assert (out[:, :320] == (255, 0, 0)).all()
    assert (out[:, 320:] == base[:, 320:]).all()


async def test_the_real_helper_uses_the_displayed_orientation_of_a_jpeg(
    env: dict[str, Path], workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pil = pytest.importorskip("PIL.Image")
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", Path(__file__).resolve().parents[3] / "core" / "engine" / "composite_helper.py")
    exif = pil.Exif()
    exif[0x0112] = 6
    pil.new("RGB", (60, 40), (0, 0, 255)).save(workspace / "r.jpg", "JPEG", exif=exif)
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))
    region = [{"shape": "rect", "x": 0, "y": 0, "w": 40, "h": 30}]
    result = await (await _prepared(_tool(env, workspace), ctx, image="r.jpg", region=region)).execute()
    assert not result.is_error, result.output
    assert png_size((workspace / "out.png").read_bytes()) == (40, 60)


def _huge(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims width x height (the checks read the header only)."""
    data = bytearray(png(8, 8))
    data[16:24] = width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return bytes(data)


@pytest.mark.parametrize("which", ["image", "mask"])
async def test_a_masked_call_refuses_an_input_over_50_megapixels_without_a_process(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, which: str
) -> None:
    (workspace / "huge.png").write_bytes(_huge(12000, 12000))
    args: dict[str, Any] = {"mask": "mask.png"} if which == "image" else {"mask": "huge.png"}
    if which == "image":
        args["image"] = "huge.png"
        args["mask"] = "mask.png"
    result = await _refused(tool, ctx, **args)
    assert result.summary == "invalid"
    assert "huge.png" in result.output and "50 megapixels" in result.output and "12000\u00d712000" in result.output
    assert not (tmp_path / "comp.jsonl").exists() and not (tmp_path / "mflux.json").exists()


async def test_a_region_call_refuses_a_huge_source_but_an_unmasked_call_does_not(
    tool: EditImageTool, ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "huge.png").write_bytes(_huge(12000, 12000))
    result = await _refused(tool, ctx, image="huge.png", region=REGION)
    assert "50 megapixels" in result.output
    await _prepared(tool, ctx, image="huge.png")


async def test_the_real_helper_refuses_a_tiny_selection_that_blurs_away_and_mflux_never_starts(
    env: dict[str, Path], workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pil = pytest.importorskip("PIL.Image")
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", Path(__file__).resolve().parents[3] / "core" / "engine" / "composite_helper.py")
    pil.new("RGB", (100, 100), (1, 2, 3)).save(workspace / "s.png")
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))
    region = [{"shape": "rect", "x": 50, "y": 50, "w": 1, "h": 1}]
    result = await (await _prepared(_tool(env, workspace), ctx, image="s.png", region=region, feather=64)).execute()
    assert result.is_error and "the mask selects nothing" in result.output
    assert not (tmp_path / "mflux.json").exists()


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
async def test_the_real_helper_uses_the_displayed_orientation_of_a_png_or_webp(
    env: dict[str, Path], workspace: Path, monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    pil = pytest.importorskip("PIL.Image")
    monkeypatch.setattr("core.engine.tools.image.COMPOSITE_HELPER", Path(__file__).resolve().parents[3] / "core" / "engine" / "composite_helper.py")
    exif = pil.Exif()
    exif[0x0112] = 6
    name = f"r.{fmt.lower()}"
    pil.new("RGB", (60, 40), (0, 0, 255)).save(workspace / name, fmt, exif=exif)
    ctx = ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace))
    region = [{"shape": "rect", "x": 0, "y": 50, "w": 40, "h": 10}]  # only valid in the rotated 40x60
    result = await (await _prepared(_tool(env, workspace), ctx, image=name, region=region)).execute()
    assert not result.is_error, result.output
    assert png_size((workspace / "out.png").read_bytes()) == (40, 60)


async def test_a_symlink_planted_as_the_composite_output_is_not_followed(
    tool: EditImageTool, ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim.png"
    victim.write_bytes(png(64, 48))
    monkeypatch.setenv("FAKE_COMPOSITE_PLANT", str(victim))
    monkeypatch.setenv("FAKE_COMPOSITE_MODE", "nofile")
    result = await (await _prepared(tool, ctx, region=REGION)).execute()
    assert result.is_error and "compositing wrote no image" in result.output
    assert not (workspace / "out.png").exists()


async def test_a_symlinked_mflux_output_is_refused(tmp_path: Path) -> None:
    from core.engine.tools.image import _read_png

    real = tmp_path / "real.png"
    real.write_bytes(png(8, 8))
    link = tmp_path / "image.png"
    link.symlink_to(real)
    result = await _read_png(link, (8, 8), "mflux")
    assert isinstance(result, ToolResult) and "mflux wrote no image" in result.output
    assert not isinstance(await _read_png(real, (8, 8), "mflux"), ToolResult)


async def test_a_passed_deadline_is_a_timeout_and_starts_no_helper(
    tool: EditImageTool, tmp_path: Path
) -> None:
    import time

    from core.engine.tools.image import _composite

    result = await _composite("prepare", [], tmp_path, tool.generator, time.monotonic() - 1)
    assert result is not None and result.output.startswith("timed out after")
    assert result.summary == "timed out"
    assert not (tmp_path / "comp.jsonl").exists()
