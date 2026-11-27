"""The CLI's image and media extras only forward to agent-core, which owns the Apple Silicon pins."""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _data() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_the_extras_forward_to_agent_core() -> None:
    extras = _data()["project"]["optional-dependencies"]
    assert extras["image"] == ["agent-core[image]"]
    assert extras["media"] == ["agent-core[media]"]


def test_mflux_is_not_a_default_dependency() -> None:
    assert not any("mflux" in dep for dep in _data()["project"]["dependencies"])


def test_uv_environments_keep_the_lock_resolvable() -> None:
    assert _data()["tool"]["uv"]["environments"] == ["sys_platform == 'darwin'", "sys_platform == 'linux'"]
