"""AC16: the image extra is opt-in and limited to Apple Silicon; the lock can resolve it."""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _data() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_the_image_extra_pins_mflux_for_apple_silicon_only() -> None:
    extra = _data()["project"]["optional-dependencies"]["image"]
    assert extra == ["mflux==0.24.0; sys_platform == 'darwin' and platform_machine == 'arm64'"]


def test_mflux_is_not_a_default_dependency() -> None:
    assert not any("mflux" in dep for dep in _data()["project"]["dependencies"])


def test_uv_environments_keep_the_lock_resolvable() -> None:
    assert _data()["tool"]["uv"]["environments"] == ["sys_platform == 'darwin'", "sys_platform == 'linux'"]
