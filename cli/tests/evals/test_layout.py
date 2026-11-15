"""AC2: the harness and its fixture's tests are never collected by the CLI suite."""

from __future__ import annotations

import tomllib
from pathlib import Path

CLI = Path(__file__).resolve().parents[2]


def test_pytest_only_collects_the_tests_folder() -> None:
    config = tomllib.loads((CLI / "pyproject.toml").read_text())
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_only_the_fixture_has_test_files_under_evals() -> None:
    found = [p.relative_to(CLI) for p in (CLI / "evals").rglob("test_*.py")]
    assert found
    assert all(p.parts[:3] == ("evals", "fixtures", "sample") for p in found)


def test_the_eval_target_is_not_part_of_test_or_ci_targets() -> None:
    makefile = (CLI / "makefile").read_text()
    for target in ("test", "check", "check-ci"):
        body = makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert "eval" not in body
