from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytest_plugins = ["core.testing.plugin"]

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    def resolve(name: str) -> Path:
        return FIXTURES / name

    return resolve
