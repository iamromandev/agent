"""The upscale part of imagegen: output size rule, known-model table, cache check and argv (phase A: R2, R3, R8, R9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import (
    COMMAND,
    DEFAULT_UPSCALE_MODEL,
    EDIT_COMMAND,
    EDIT_EXTRA,
    KNOWN_MODELS,
    MAX_UPSCALE_SIDE,
    UPSCALE_COMMAND,
    UPSCALE_EXTRA,
    UPSCALE_SCALES,
    ImageRequest,
    MfluxGenerator,
    model_cached,
    resolve_model,
    upscale_size,
)

FILES = ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")
FILES_7B = ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors")
REPO = "numz/SeedVR2_comfyUI"
REVISION = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"


@pytest.mark.parametrize(
    "size, scale, expected",
    [
        ((333, 250), 2, (660, 496)),
        ((300, 201), 2, (596, 400)),
        ((250, 333), 2, (496, 660)),
        ((517, 517), 2, (1024, 1024)),
        ((1000, 750), 2, (1984, 1488)),
        ((512, 512), 2, (1024, 1024)),
        ((1024, 1024), 2, (2048, 2048)),
        ((256, 256), 4, (1024, 1024)),
    ],
)
def test_upscale_size_follows_mfluxs_rule(size: tuple[int, int], scale: int, expected: tuple[int, int]) -> None:
    assert upscale_size(*size, scale) == expected


def test_the_constants_are_the_spiked_ones() -> None:
    assert UPSCALE_COMMAND == "mflux-upscale-seedvr2"
    assert DEFAULT_UPSCALE_MODEL == "seedvr2-3b"
    assert UPSCALE_EXTRA == ("--low-ram", "--quantize", "8")
    assert UPSCALE_SCALES == (2, 3, 4)
    assert MAX_UPSCALE_SIDE == 2048


def test_known_aliases_map_to_the_repo_and_its_two_files() -> None:
    assert set(KNOWN_MODELS) == {"seedvr2-3b", "seedvr2", "seedvr2-7b"}
    assert resolve_model("seedvr2-3b") == (REPO, FILES)
    assert resolve_model("seedvr2") == (REPO, FILES)


def test_the_7b_alias_maps_to_its_own_weights() -> None:
    assert resolve_model("seedvr2-7b") == (REPO, FILES_7B)


@pytest.mark.parametrize("name", ["SeedVR2-3B", "SEEDVR2", "seedvr2-7B"])
def test_alias_lookup_ignores_case_like_mflux(name: str) -> None:
    assert resolve_model(name)[0] == REPO
    assert resolve_model(name)[1] is not None


def test_the_known_entries_pin_mfluxs_revision() -> None:
    assert {known.revision for known in KNOWN_MODELS.values()} == {REVISION}


def test_unknown_names_resolve_to_themselves_without_files() -> None:
    assert resolve_model("a/b") == ("a/b", None)


def _snapshot(hub: Path, names: list[str], nested: bool = True, rev: str = REVISION) -> Path:
    """The new shared-store layout: snapshot link -> repo blobs link -> hub-level blobs/<2>/<hash>."""
    repo = hub / f"models--{REPO.replace('/', '--')}"
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name in names:
        store = hub / "blobs" / "ab"
        store.mkdir(parents=True, exist_ok=True)
        (store / name).write_bytes(b"x" * 10)
        if nested:
            (repo / "blobs" / name).symlink_to(Path("..") / ".." / "blobs" / "ab" / name)
            (snap / name).symlink_to(Path("..") / ".." / "blobs" / name)
        else:
            (snap / name).write_bytes(b"x" * 10)
    return repo


def test_a_known_model_is_cached_when_both_files_resolve_in_the_shared_store(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", list(FILES))
    assert model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_a_known_model_is_not_cached_with_only_the_vae(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", [FILES[1]])
    assert not model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_other_files_of_the_repo_do_not_make_a_known_model_cached(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", ["README.md"])
    assert not model_cached("seedvr2", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_a_dangling_link_is_not_a_present_file(tmp_path: Path) -> None:
    repo = _snapshot(tmp_path / "hub", list(FILES))
    for blob in (tmp_path / "hub" / "blobs" / "ab").iterdir():
        blob.unlink()
    assert (repo / "snapshots" / REVISION / FILES[0]).is_symlink()
    assert not model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_files_only_in_another_revision_are_not_cached(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", list(FILES), rev="other")
    assert not model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_the_pinned_revision_counts_even_beside_another_one(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", list(FILES))
    (tmp_path / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / "other").mkdir()
    assert model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})


def test_a_folder_named_like_the_alias_does_not_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "seedvr2-3b").mkdir()
    monkeypatch.chdir(tmp_path)
    assert not model_cached("seedvr2-3b", {"HF_HUB_CACHE": str(tmp_path / "hub")})
    assert MfluxGenerator(tmp_path / "mflux", "seedvr2-3b", 5, Guardrails.off(tmp_path)).model == "seedvr2-3b"


def test_the_repo_id_of_a_known_model_still_uses_the_snapshot_rule(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hub", [FILES[1]])
    assert model_cached(REPO, {"HF_HUB_CACHE": str(tmp_path / "hub")})


def _generator(tmp_path: Path, **kwargs: object) -> MfluxGenerator:
    return MfluxGenerator(tmp_path / "mflux", "m/x", 5, Guardrails.off(tmp_path), **kwargs)  # ty: ignore[invalid-argument-type]


def test_the_upscale_argv_is_one_item_per_value(tmp_path: Path) -> None:
    generator = _generator(tmp_path, extra=UPSCALE_EXTRA, kind="upscale")
    request = ImageRequest("", 1024, 1024, 1, 7, (tmp_path / "in-0.png",), scale=2, softness=0.5)
    out = tmp_path / "o"
    assert generator.argv(request, out) == [
        str(tmp_path / "mflux"),
        "--model", "m/x",
        "--image-path", str(tmp_path / "in-0.png"),
        "--resolution", "2x",
        "--seed", "7",
        "--softness", "0.5",
        "--low-ram", "--quantize", "8",
        "--output", str(out / "image.png"),
        "--no-exif",
    ]  # fmt: skip


def test_a_zero_softness_is_written_as_0(tmp_path: Path) -> None:
    generator = _generator(tmp_path, extra=UPSCALE_EXTRA, kind="upscale")
    request = ImageRequest("", 128, 128, 1, 0, (tmp_path / "in-0.png",), scale=4)
    argv = generator.argv(request, tmp_path)
    assert argv[argv.index("--softness") + 1] == "0"
    assert argv[argv.index("--resolution") + 1] == "4x"


def test_generate_and_edit_argv_stay_byte_identical(tmp_path: Path) -> None:
    gen = _generator(tmp_path)
    assert gen.argv(ImageRequest("a cat", 512, 256, 9, 3), tmp_path) == [
        str(tmp_path / "mflux"), "--model", "m/x", "--prompt=a cat", "--width", "512", "--height", "256",
        "--steps", "9", "--seed", "3", "--output", str(tmp_path / "image.png"), "--no-exif",
    ]  # fmt: skip
    edit = _generator(tmp_path, extra=EDIT_EXTRA)
    request = ImageRequest("fix", 512, 256, 4, 3, (tmp_path / "in-0.png", tmp_path / "in-1.png"))
    assert edit.argv(request, tmp_path) == [
        str(tmp_path / "mflux"), "--model", "m/x", "--prompt=fix",
        "--image-paths", str(tmp_path / "in-0.png"), str(tmp_path / "in-1.png"),
        "--width", "512", "--height", "256", "--steps", "4", "--seed", "3", "--quantize", "4",
        "--output", str(tmp_path / "image.png"), "--no-exif",
    ]  # fmt: skip


def test_unused_commands_stay_distinct() -> None:
    assert len({COMMAND, EDIT_COMMAND, UPSCALE_COMMAND}) == 3
