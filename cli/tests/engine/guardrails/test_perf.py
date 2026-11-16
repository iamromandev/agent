"""NFR3: what the guardrails add per command. The sandbox start-up test needs macOS."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
from src.engine.guardrails import sandbox as sandbox_module
from src.engine.guardrails.config import GuardConfig
from src.engine.guardrails.paths import ProtectedPaths
from src.engine.guardrails.sandbox import SANDBOX_EXEC
from src.engine.guardrails.setup import prepare

SANDBOX_START_LIMIT_MS = 50.0  # NFR3: added per bash command, median
RUNS = 25


def _median_ms(argv: list[str]) -> float:
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


@pytest.mark.skipif(sys.platform != "darwin" or not SANDBOX_EXEC.exists(), reason="needs macOS and /usr/bin/sandbox-exec")
def test_the_sandbox_adds_at_most_50_ms_to_the_median_command_start(tmp_path: Path) -> None:
    home = (tmp_path / "home").resolve()
    home.mkdir()
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    guard = prepare(workspace, None, None, {}, "Darwin", home=home)
    assert guard.sandbox.profile is not None, guard.warnings
    plain = ["/usr/bin/true"]
    _median_ms(plain)  # warm the file cache

    bare = _median_ms(plain)
    wrapped = _median_ms(guard.sandbox.wrap(plain))

    print(f"NFR3 sandbox start: bare={bare:.1f}ms sandboxed={wrapped:.1f}ms added={wrapped - bare:.1f}ms")
    assert wrapped - bare <= SANDBOX_START_LIMIT_MS


def test_building_the_profile_with_many_settings_patterns_is_fast(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    patterns = tuple(f"secrets{index}/**" for index in range(100)) + tuple(f"*.tok{index}" for index in range(100))
    config = GuardConfig(extra_protected=patterns)
    start = time.perf_counter()
    sandbox_module.build_profile(
        workspace, [workspace], [], ProtectedPaths(home, config.extra_protected), True
    )
    elapsed = time.perf_counter() - start

    print(f"profile with 200 patterns: {elapsed * 1000:.1f}ms")
    assert elapsed < 0.5
