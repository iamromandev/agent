"""`evals/pixel_compare.py`: the `same`, `changed` and `varied` modes (Pillow and numpy are loaded by importlib)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")
SCRIPT = Path(__file__).resolve().parents[2] / "evals" / "pixel_compare.py"
Image = importlib.import_module("PIL.Image")


def _save(path: Path, left: int, right: int, size: tuple[int, int] = (16, 4)) -> Path:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (left if x < 8 else right, 0, 0))
    image.save(path)
    return path


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_same_passes_when_the_right_is_identical_even_if_the_left_differs(tmp_path: Path) -> None:
    src, out = _save(tmp_path / "s.png", 10, 50), _save(tmp_path / "o.png", 99, 50)
    assert _run("same", src, out, 8).returncode == 0


def test_same_fails_with_one_value_off_on_the_right(tmp_path: Path) -> None:
    src, out = _save(tmp_path / "s.png", 10, 50), _save(tmp_path / "o.png", 10, 51)
    done = _run("same", src, out, 8)
    assert done.returncode == 1 and done.stderr.strip()


def test_same_fails_on_a_different_size(tmp_path: Path) -> None:
    src, out = _save(tmp_path / "s.png", 10, 50), _save(tmp_path / "o.png", 10, 50, (20, 4))
    assert _run("same", src, out, 8).returncode == 1


def test_changed_passes_with_enough_differing_pixels_on_the_left(tmp_path: Path) -> None:
    src, out = _save(tmp_path / "s.png", 10, 50), _save(tmp_path / "o.png", 99, 50)
    assert _run("changed", src, out, 8, 32).returncode == 0  # 8 columns x 4 rows


def test_changed_fails_when_the_output_is_the_source(tmp_path: Path) -> None:
    src = _save(tmp_path / "s.png", 10, 50)
    done = _run("changed", src, src, 8, 1)
    assert done.returncode == 1 and "expected at least" in done.stderr


def test_changed_fails_below_the_minimum_and_ignores_the_right(tmp_path: Path) -> None:
    src, out = _save(tmp_path / "s.png", 10, 50), _save(tmp_path / "o.png", 99, 77)
    assert _run("changed", src, out, 8, 33).returncode == 1


def test_varied_passes_for_two_colours_and_fails_for_one(tmp_path: Path) -> None:
    assert _run("varied", _save(tmp_path / "a.png", 10, 50)).returncode == 0
    flat = _run("varied", _save(tmp_path / "b.png", 7, 7))
    assert flat.returncode == 1 and "flat" in flat.stderr


def test_an_unreadable_input_is_exit_three_not_a_failed_check(tmp_path: Path) -> None:
    good = _save(tmp_path / "s.png", 10, 50)
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    for args in (("same", good, bad, 8), ("changed", bad, good, 8, 1), ("varied", bad), ("varied", tmp_path / "none.png")):
        done = _run(*args)
        assert done.returncode == 3, args
        assert len(done.stderr.strip().splitlines()) == 1


def test_bad_usage_exits_two(tmp_path: Path) -> None:
    for args in ((), ("same", tmp_path / "s.png"), ("nope", "a", "b", "1"), ("same", "a", "b", "x"), ("changed", "a", "b", "1")):
        assert _run(*args).returncode == 2, args
