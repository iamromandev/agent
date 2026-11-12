from __future__ import annotations

import pytest
from src.core.runtime import format_uptime


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (59, "59s"),
        (60, "1m"),
        (3661, "1h 1m 1s"),
        (90061, "1d 1h 1m 1s"),
    ],
)
def test_format_uptime(seconds: float, expected: str) -> None:
    assert format_uptime(seconds) == expected
