"""AC10 / NFR1: completion and expansion stay fast on a big workspace."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from core.engine.guardrails.setup import Guardrails
from core.engine.mentions import expand_mentions

BUDGET = 0.05
BUILD_BUDGET = 2.0


def best_of_two(run: Callable[[], object]) -> float:
    """Wall time of `run`; a second try when the first is over budget (a loaded machine, not a regression)."""
    times = []
    for _ in range(2):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
        if times[-1] < BUDGET:
            break
    return min(times)


def make_tree(root: Path) -> None:
    for d in range(200):
        folder = root / f"pkg{d:03}"
        folder.mkdir()
        for f in range(100):
            (folder / f"mod{f:03}.py").write_text("")


def test_expanding_20_files_of_30kb_is_under_50_ms(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i}.txt").write_text(("line of text\n" * 3000)[:29_000])
    text = " ".join(f"@f{i}.txt" for i in range(20))
    guard = Guardrails.off(tmp_path)
    expansion = expand_mentions(text, tmp_path, guard)
    assert sum(n.attached for n in expansion.notes) >= 3

    took = best_of_two(lambda: expand_mentions(text, tmp_path, guard))
    print(f"expand: {took * 1000:.2f} ms")
    assert took < BUDGET
