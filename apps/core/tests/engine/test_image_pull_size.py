"""The download size from `hf download --dry-run` and the free-space check (R9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.engine import image_pull
from core.engine.guardrails.setup import Guardrails
from core.engine.image_pull import PullSizeError, download_size, free_space, parse_dry_run
from core.testing.image_fakes import dry_run, fake_hf

SAMPLE_A = dry_run(
    [
        (".gitattributes", "1.6K"),
        ("README.md", "10.0K"),
        ("editing.jpg", "2.5M"),
        ("flux-2-klein-4b.safetensors", "7.8G"),
        ("model_index.json", "446.0"),
        ("scheduler/scheduler_config.json", "486.0"),
        ("text_encoder/config.json", "-"),
        ("text_encoder/model-00001-of-00002.safetensors", "-"),
        ("tokenizer/vocab.json", "-"),
        ("transformer/diffusion_pytorch_model.safetensors", "-"),
        ("vae/config.json", "-"),
    ]
)

SAMPLE_B = dry_run(
    [
        (".DS_Store", "6.1K"),
        ("tokenizer/added_tokens.json", "707.0"),
        ("tokenizer/chat_template.jinja", "4.2K"),
        ("tokenizer/merges.txt", "1.7M"),
        ("tokenizer/tokenizer.json", "11.4M"),
        ("transformer/0.safetensors", "2.1G"),
        ("transformer/1.safetensors", "1.3G"),
        ("transformer/model.safetensors.index.json", "61.6K"),
        ("vae/0.safetensors", "164.7M"),
        ("z_image_example.png", "1.2M"),
    ],
    "21 files (out of 21) totalling 5.9G",
)


def test_cached_rows_and_root_files_do_not_count() -> None:
    assert parse_dry_run(SAMPLE_A) == 486


def test_units_are_decimal_and_only_subfolders_count() -> None:
    expected = 707 + 4200 + 1_700_000 + 11_400_000 + 2_100_000_000 + 1_300_000_000 + 61_600 + 164_700_000
    assert parse_dry_run(SAMPLE_B) == expected


def test_a_t_unit_is_read() -> None:
    assert parse_dry_run(dry_run([("transformer/a", "1.5T")])) == 1_500_000_000_000


@pytest.mark.parametrize("text", ["", "nothing here\n", "[dry-run] Will download 0 files.\n"])
def test_no_table_is_an_error(text: str) -> None:
    with pytest.raises(PullSizeError):
        parse_dry_run(text)


def test_a_bad_size_is_an_error() -> None:
    with pytest.raises(PullSizeError):
        parse_dry_run(dry_run([("transformer/a", "lots")]))


def test_a_trailing_url_line_is_not_a_row() -> None:
    assert parse_dry_run(SAMPLE_A + "see https://x/y\n") == 486


def test_a_row_without_a_tab_is_skipped() -> None:
    assert parse_dry_run("file\tsize\na/b\ntransformer/c\t1.0K\n") == 1000


def test_a_row_with_an_empty_size_is_skipped() -> None:
    assert parse_dry_run("file\tsize\na/b\t\ntransformer/c\t1.0K\n") == 1000


def test_free_space_uses_the_nearest_existing_parent(tmp_path: Path) -> None:
    free, folder = free_space(tmp_path / "a" / "b")
    assert folder == tmp_path
    assert free > 0


async def test_download_size_runs_hf_with_its_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    hf = fake_hf(tmp_path, SAMPLE_A)
    assert await download_size(hf, "org/name", Guardrails.off(tmp_path)) == 486
    assert json.loads(dump.read_text())["argv"] == ["download", "org/name", "--dry-run"]


async def test_download_size_drops_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = Guardrails.off(tmp_path)
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    await download_size(fake_hf(tmp_path, SAMPLE_A), "a/b", guard)
    assert "HF_TOKEN" not in json.loads(dump.read_text())["env"]


async def test_a_failing_hf_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PullSizeError, match="hf exited 3"):
        await download_size(fake_hf(tmp_path, "", code=3), "a/b", Guardrails.off(tmp_path))


async def test_a_slow_hf_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_pull, "SIZE_TIMEOUT", 0.3)
    with pytest.raises(PullSizeError, match="timed out"):
        await download_size(fake_hf(tmp_path, SAMPLE_A, sleep=30), "a/b", Guardrails.off(tmp_path))
