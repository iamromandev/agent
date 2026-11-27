"""Input images for edit_image: header sizes per format, workspace and paste resolution, staging (AC2, AC3, AC13)."""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import pytest
from core.engine.guardrails.setup import Guardrails, prepare
from core.engine.image_inputs import InputImage, default_size, image_dimensions, resolve_input, stage
from core.engine.provider.base import Image
from core.engine.tools.base import ToolContext, ToolResult
from core.testing.image_fakes import png


def jpeg(width: int, height: int, *, junk_first: bool = True) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0" + b"\0" * 9
    dht = b"\xff\xc4" + struct.pack(">H", 4) + b"\0\0"  # 0xC4 is not a frame header
    sof = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + app0 + (dht if junk_first else b"") + sof + b"\xff\xd9"


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\0\0\0;"


def _riff(chunk: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_lossy(width: int, height: int) -> bytes:
    body = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\0" * 4
    return _riff(b"VP8 " + struct.pack("<I", len(body)) + body)


def webp_lossless(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    body = b"\x2f" + struct.pack("<I", bits) + b"\0" * 4
    return _riff(b"VP8L" + struct.pack("<I", len(body)) + body)


def webp_extended(width: int, height: int) -> bytes:
    body = b"\0\0\0\0" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return _riff(b"VP8X" + struct.pack("<I", len(body)) + body)


@pytest.mark.parametrize(
    "fmt, data",
    [
        ("png", png(30, 20)),
        ("jpeg", jpeg(30, 20)),
        ("jpeg", jpeg(30, 20, junk_first=False)),
        ("gif", gif(30, 20)),
        ("webp", webp_lossy(30, 20)),
        ("webp", webp_lossless(30, 20)),
        ("webp", webp_extended(30, 20)),
    ],
)
def test_image_dimensions_read_the_header_of_each_format(fmt: str, data: bytes) -> None:
    assert image_dimensions(fmt, data) == (30, 20)  # ty: ignore[invalid-argument-type]


def test_image_dimensions_are_none_for_truncated_or_empty_headers() -> None:
    cases = [
        ("png", png(30, 20)[:20]),
        ("jpeg", jpeg(30, 20)[:10]),
        ("jpeg", b"\xff\xd8\xff"),
        ("gif", b"GIF89a\x01"),
        ("webp", webp_lossy(30, 20)[:24]),
        ("webp", webp_lossless(30, 20)[:22]),
        ("webp", webp_extended(30, 20)[:26]),
        ("png", b""),
    ]
    for fmt, data in cases:
        assert image_dimensions(fmt, data) is None, (fmt, data[:8])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "source, expected",
    [
        ((1000, 750), (1008, 752)),
        ((4000, 3000), (1024, 768)),
        ((4000, 1000), (1024, 256)),
        ((1024, 1024), (1024, 1024)),
        ((10, 10), (256, 256)),
        ((500, 500), (496, 496)),
    ],
)
def test_default_size_keeps_the_aspect_ratio_caps_the_long_side_at_1024_and_never_scales_up(
    source: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert default_size(*source) == expected


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def _ctx(workspace: Path, *pasted: Image) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace), pasted=pasted)


def _write(workspace: Path, name: str, data: bytes) -> Path:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _ok(result: InputImage | ToolResult) -> InputImage:
    assert isinstance(result, InputImage), result
    return result


def test_a_workspace_png_resolves_with_its_size_and_label(workspace: Path) -> None:
    _write(workspace, "a/hero.png", png(40, 30))
    got = _ok(resolve_input("a/hero.png", _ctx(workspace)))
    assert (got.label, got.fmt, got.width, got.height) == ("a/hero.png", "png", 40, 30)
    assert got.data == png(40, 30)


def test_a_jpeg_source_is_accepted_whatever_its_name(workspace: Path) -> None:
    _write(workspace, "photo.png", jpeg(64, 48))
    got = _ok(resolve_input("photo.png", _ctx(workspace)))
    assert (got.fmt, got.width, got.height) == ("jpeg", 64, 48)


def test_a_paste_label_resolves_to_the_pasted_bytes(workspace: Path) -> None:
    pasted = Image("png", png(8, 6), "clipboard")
    got = _ok(resolve_input("clipboard:1", _ctx(workspace, pasted)))
    assert (got.label, got.data, got.width, got.height) == ("clipboard:1", pasted.data, 8, 6)
    assert got.identity == ("clipboard", 1)


def test_markup_in_a_label_is_written_out(workspace: Path) -> None:
    _write(workspace, "[bold]x[/].png", png(8, 8))
    assert _ok(resolve_input("[bold]x[/].png", _ctx(workspace))).label == "[bold]x[/].png"
    refused = resolve_input("nope\x1b[2J.png", _ctx(workspace))
    assert isinstance(refused, ToolResult) and "\x1b" not in refused.output


def _refusals(workspace: Path, tmp_path: Path) -> list[tuple[str, ToolContext, str, str]]:
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(8, 8))
    (workspace / "link.png").symlink_to(outside)
    _write(workspace, ".env", b"SECRET=1\n")
    _write(workspace, "x.png", b"just text, not an image")
    _write(workspace, "big.png", png(8, 8) + b"\0" * (6 * 1024 * 1024))
    (workspace / "dir").mkdir()
    one = _ctx(workspace, Image("png", png(8, 8), "clipboard"))
    none = _ctx(workspace)
    return [
        (str(outside), none, "refused: outside workspace", "outside"),
        ("link.png", none, "refused: outside workspace", "symlink to outside"),
        (".env", none, "refused: protected path", "protected"),
        ("x.png", none, "not an image", "text file named .png"),
        ("big.png", none, "image too large", "6 MB"),
        ("dir", none, "directory", "folder"),
        ("clipboard:2", one, "no such paste", "missing label"),
        ("clipboard:1", none, "no such paste", "no paste"),
        ("clipboard:x", one, "no such paste", "bad label"),
        ("missing.png", none, "not found", "no file"),
    ]


def test_every_refusal_is_an_error_with_its_summary(workspace: Path, tmp_path: Path) -> None:
    for ref, ctx, summary, why in _refusals(workspace, tmp_path):
        result = resolve_input(ref, ctx)
        assert isinstance(result, ToolResult) and result.is_error, why
        assert result.summary == summary, why


def test_a_missing_paste_label_lists_the_ones_that_exist(workspace: Path) -> None:
    ctx = _ctx(workspace, Image("png", png(8, 8), "clipboard"))
    result = resolve_input("clipboard:2", ctx)
    assert isinstance(result, ToolResult) and "clipboard:1" in result.output
    nothing = resolve_input("clipboard:1", _ctx(workspace))
    assert isinstance(nothing, ToolResult) and "no pasted images" in nothing.output


def test_the_same_file_has_the_same_identity_under_two_spellings(workspace: Path) -> None:
    _write(workspace, "a.png", png(8, 8))
    (workspace / "b.png").symlink_to(workspace / "a.png")
    first = _ok(resolve_input("a.png", _ctx(workspace)))
    second = _ok(resolve_input("./b.png", _ctx(workspace)))
    assert first.identity == second.identity


def test_stage_writes_private_copies_of_the_checked_bytes(workspace: Path, tmp_path: Path) -> None:
    folder = tmp_path / "stage"
    folder.mkdir()
    items = [
        InputImage("a", "png", png(8, 8), 8, 8, ("x", 1)),
        InputImage("b", "jpeg", jpeg(8, 8), 8, 8, ("x", 2)),
        InputImage("c", "webp", webp_lossy(8, 8), 8, 8, ("x", 3)),
    ]
    paths = stage(items, folder)
    assert [p.name for p in paths] == ["in-0.png", "in-1.jpg", "in-2.webp"]
    assert all(p.parent == folder for p in paths)
    assert [p.read_bytes() for p in paths] == [i.data for i in items]
    assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in paths)


def test_checking_and_staging_three_5_mb_inputs_is_fast(workspace: Path, tmp_path: Path) -> None:
    data = png(8, 8) + b"\0" * (5 * 1024 * 1024 - len(png(8, 8)))
    for name in ("a.png", "b.png", "c.png"):
        _write(workspace, name, data)
    folder = tmp_path / "stage"
    folder.mkdir()
    ctx = _ctx(workspace)
    start = time.perf_counter()
    inputs = [_ok(resolve_input(name, ctx)) for name in ("a.png", "b.png", "c.png")]
    stage(inputs, folder)
    assert time.perf_counter() - start < 0.2


def _guarded(workspace: Path, tmp_path: Path) -> ToolContext:
    """A context with a real guard, built after the files exist so the workspace scan records their identity."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    guard = prepare(workspace, None, None, {}, home=home)
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=guard)


def _refused(result: InputImage | ToolResult) -> ToolResult:
    assert isinstance(result, ToolResult) and result.is_error, result
    return result


def test_a_hard_link_to_a_protected_file_is_refused(workspace: Path, tmp_path: Path) -> None:
    _write(workspace, ".env", png(8, 8))
    os.link(workspace / ".env", workspace / "link.png")
    assert _refused(resolve_input("link.png", _guarded(workspace, tmp_path))).output == "protected path"


def test_a_symlink_to_a_protected_file_is_refused(workspace: Path, tmp_path: Path) -> None:
    _write(workspace, ".env", png(8, 8))
    (workspace / "link.png").symlink_to(workspace / ".env")
    assert _refuse_text(resolve_input("link.png", _guarded(workspace, tmp_path)))


def _refuse_text(result: InputImage | ToolResult) -> str:
    return _refused(result).output


def test_a_fifo_is_not_a_regular_file_and_does_not_block(workspace: Path, tmp_path: Path) -> None:
    os.mkfifo(workspace / "f.png")
    started = time.monotonic()
    result = _refused(resolve_input("f.png", _guarded(workspace, tmp_path)))
    assert "not a regular file" in result.output and time.monotonic() - started < 2


def test_a_path_swapped_for_a_protected_file_after_the_check_is_refused_by_the_fd_check(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(workspace, ".env", png(8, 8))
    _write(workspace, "a.png", png(8, 8))
    ctx = _guarded(workspace, tmp_path)
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda p, flags, *a, **k: real_open(workspace / ".env", flags, *a, **k) if str(p).endswith("a.png") else real_open(p, flags, *a, **k))
    assert _refused(resolve_input("a.png", ctx)).output == "protected path"


def test_a_path_swapped_for_a_file_outside_the_workspace_after_the_check_is_refused(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(8, 8))
    _write(workspace, "a.png", png(8, 8))
    ctx = _guarded(workspace, tmp_path)
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda p, flags, *a, **k: real_open(outside, flags, *a, **k) if str(p).endswith("a.png") else real_open(p, flags, *a, **k))
    assert "outside workspace" in _refused(resolve_input("a.png", ctx)).output


def test_a_corrupt_jpeg_without_a_frame_header_is_not_scanned_to_the_end() -> None:
    data = b"\xff\xd8" + b"\xff\xda" + struct.pack(">H", 2) + b"\0" * 5_000_000
    started = time.monotonic()
    assert image_dimensions("jpeg", data) is None
    assert time.monotonic() - started < 0.1
