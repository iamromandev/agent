"""The upscale role of `/images`: status from the known files, the dry-run rule and the warm-up (phase A: R9-R11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.engine.guardrails.setup import Guardrails
from core.engine.image_pull import ROLES, ImagePuller, Role, free_space, model_status, parse_dry_run
from core.engine.imagegen import EDIT_EXTRA, UPSCALE_EXTRA, MfluxGenerator
from core.testing.image_fakes import dry_run, fake_hf, fake_mflux

from tests.engine.test_image_pull import Notes, _cache, _ignore

FILES = ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")
REPO_DIR = "models--numz--SeedVR2_comfyUI"


def _env(tmp_path: Path) -> dict[str, str]:
    return {"HF_HUB_CACHE": str(tmp_path / "hub")}


REVISION = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"


def _nested(tmp_path: Path, sizes: dict[str, int], incomplete: str | None = None, rev: str = REVISION) -> None:
    """The shared-store layout: snapshot link -> repo blobs link -> hub-level blobs/<2>/<hash>."""
    hub = tmp_path / "hub"
    repo = hub / REPO_DIR
    (repo / "snapshots" / rev).mkdir(parents=True)
    (repo / "blobs").mkdir()
    (hub / "blobs" / "ab").mkdir(parents=True)
    for name, size in sizes.items():
        (hub / "blobs" / "ab" / name).write_bytes(b"x" * size)
        (repo / "blobs" / name).symlink_to(Path("..") / ".." / "blobs" / "ab" / name)
        (repo / "snapshots" / rev / name).symlink_to(Path("..") / ".." / "blobs" / name)
    if incomplete:
        (repo / "blobs" / incomplete).write_bytes(b"y")


def test_the_roles_end_with_upscale() -> None:
    assert ROLES == ("generate", "edit", "upscale")


def test_a_known_model_is_cached_with_the_size_of_its_two_files_in_the_nested_layout(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90})
    status = model_status("upscale", "seedvr2-3b", _env(tmp_path))
    assert (status.state, status.size, status.model, status.role) == ("cached", 790, "seedvr2-3b", "upscale")


def test_other_files_in_the_repo_do_not_count_toward_the_size(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90, "other.bin": 5000})
    assert model_status("upscale", "seedvr2", _env(tmp_path)).size == 790


def test_only_the_vae_is_partial(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[1]: 90})
    status = model_status("upscale", "seedvr2-3b", _env(tmp_path))
    assert (status.state, status.size) == ("partial", 90)


def test_nothing_downloaded_is_missing(tmp_path: Path) -> None:
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)) == model_status("upscale", "seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "x")})
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "missing"


def test_an_unfinished_download_in_the_repo_is_partial_even_with_both_files(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90}, incomplete=f"{FILES[0]}.incomplete")
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "partial"


def test_an_unrelated_stale_incomplete_does_not_make_a_complete_model_partial(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90}, incomplete="zz.incomplete")
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "cached"


def test_a_missing_file_with_an_incomplete_download_is_partial(tmp_path: Path) -> None:
    _nested(tmp_path, {}, incomplete="zz.incomplete")
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "partial"


def test_only_another_revision_holding_the_files_is_missing(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90}, rev="other")
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "missing"


def test_a_folder_named_like_the_alias_does_not_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "seedvr2-3b").mkdir()
    monkeypatch.chdir(tmp_path)
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "missing"


def test_a_dangling_link_is_partial(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90})
    next((tmp_path / "hub" / "blobs" / "ab").iterdir()).unlink()
    assert model_status("upscale", "seedvr2-3b", _env(tmp_path)).state == "partial"


def test_the_repo_id_keeps_the_generic_status_rule(tmp_path: Path) -> None:
    _cache(tmp_path, "org/up")
    assert model_status("upscale", "org/up", {"HF_HUB_CACHE": str(tmp_path / "hub")}).state == "cached"


def test_with_files_only_the_listed_rows_count_and_root_files_too() -> None:
    text = dry_run(
        [
            ("seedvr2_ema_3b_fp16.safetensors", "6.8G"),
            ("ema_vae_fp16.safetensors", "500.0M"),
            ("seedvr2_ema_7b_fp16.safetensors", "16.5G"),
            ("sub/other.bin", "1.0G"),
            ("README.md", "10.0K"),
        ]
    )
    assert parse_dry_run(text, FILES) == 7_300_000_000
    assert parse_dry_run(text) == 1_000_000_000  # unchanged: the subfolder rule


def test_with_files_a_cached_row_counts_nothing() -> None:
    text = dry_run([(FILES[0], "-"), (FILES[1], "500.0M")])
    assert parse_dry_run(text, FILES) == 500_000_000


def _puller(tmp_path: Path, hf: Path | None = None, disk: object = None, upscale: bool = True) -> ImagePuller:
    guard = Guardrails.off(tmp_path)
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    generators: dict[Role, MfluxGenerator | None] = {
        "generate": MfluxGenerator(fake_mflux(folder, "gen"), "org/gen", 5, guard),
        "edit": MfluxGenerator(fake_mflux(folder, "edit"), "org/edit", 5, guard, extra=EDIT_EXTRA),
        "upscale": (
            MfluxGenerator(fake_mflux(folder, "up"), "seedvr2-3b", 5, guard, extra=UPSCALE_EXTRA, kind="upscale")
            if upscale
            else None
        ),
    }
    return ImagePuller(generators, hf, guard, env=_env(tmp_path), disk=disk or free_space)  # ty: ignore[invalid-argument-type]


def test_the_status_has_a_row_per_role_with_upscale_last(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 700, FILES[1]: 90})
    rows = _puller(tmp_path).status()
    assert [r.role for r in rows] == ["generate", "edit", "upscale"]
    assert (rows[2].state, rows[2].size) == ("cached", 790)
    assert _puller(tmp_path, upscale=False).status()[2].state == "unavailable"


async def test_pull_upscale_warms_only_it_with_a_64_px_grey_source_at_2x(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    notes = Notes()
    await _puller(tmp_path).pull(("upscale",), notes.emit, _ignore)
    data = json.loads(dump.read_text())
    argv = data["argv"]
    assert argv[argv.index("--model") + 1] == "seedvr2-3b"  # the alias goes to mflux unchanged
    assert argv[argv.index("--resolution") + 1] == "2x" and argv[argv.index("--seed") + 1] == "0"
    assert argv[argv.index("--softness") + 1] == "0"
    assert argv[argv.index("--low-ram") : argv.index("--low-ram") + 3] == ["--low-ram", "--quantize", "8"]
    (source,) = data["inputs"]
    assert source.endswith("in-0.png") and data["shas"][source] is not None
    assert not Path(data["cwd"]).exists()
    assert [t.split()[0] for t in notes.texts() if "ready" in t] == ["upscale"]
    assert not [t for t in notes.texts() if "failed" in t]


async def test_a_full_pull_warms_all_three_in_order(tmp_path: Path) -> None:
    for repo in ("org/gen", "org/edit"):
        _cache(tmp_path, repo)
    notes = Notes()
    await _puller(tmp_path).pull(ROLES, notes.emit, _ignore)
    assert [t.split()[0] for t in notes.texts() if "model ready" in t] == ["generate", "edit", "upscale"]


async def test_a_missing_upscale_command_is_a_failure_notice(tmp_path: Path) -> None:
    notes = Notes()
    await _puller(tmp_path, upscale=False).pull(("upscale",), notes.emit, _ignore)
    assert notes.texts() == ["upscale model pull failed: mflux command not found"]


async def test_the_disk_check_sums_the_two_files_and_asks_hf_for_the_repo_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "hfdump.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    rows = [
        (FILES[0], "6.8G"),
        (FILES[1], "500.0M"),
        ("seedvr2_ema_7b_fp16.safetensors", "16.5G"),
        ("seedvr2_ema_7b_sharp_fp16.safetensors", "16.5G"),
        ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "3.4G"),
        ("README.md", "1.0K"),
    ]
    hf = fake_hf(tmp_path, dry_run(rows, "6 files totalling 60G"))
    notes = Notes()
    puller = _puller(tmp_path, hf=hf, disk=lambda folder: (50_000_000_000, folder))
    await puller.pull(("upscale",), notes.emit, _ignore)
    assert json.loads(dump.read_text())["argv"][:2] == ["download", "numz/SeedVR2_comfyUI"]
    assert "pulling upscale model seedvr2-3b (download size 7.3 GB, 50.0 GB free)" in notes.texts()


async def test_a_cached_known_model_adds_no_size_note_and_never_refuses(tmp_path: Path) -> None:
    _nested(tmp_path, {FILES[0]: 7, FILES[1]: 9})
    hf = fake_hf(tmp_path, dry_run([(FILES[0], "-"), (FILES[1], "-"), ("big.bin", "60.0G")]))
    notes = Notes()
    await _puller(tmp_path, hf=hf, disk=lambda folder: (1, folder)).pull(("upscale",), notes.emit, _ignore)
    assert "pulling upscale model seedvr2-3b" in notes.texts()
    assert [t for t in notes.texts() if "not enough disk" in t] == []
