"""`/format`: the text of `info.format_summary`."""

from __future__ import annotations

from pathlib import Path

from core.engine.format.config import FormatSetup, FormatterSpec
from core.engine.format.runner import FormatHub
from core.engine.guardrails.setup import Guardrails
from core.info import format_summary


def hub(*specs: FormatterSpec, missing: tuple[str, ...] = (), warnings: tuple[str, ...] = (), trusted: bool = True) -> FormatHub:
    return FormatHub(
        FormatSetup(True, specs, missing, warnings), Path("."), Guardrails.off(Path(".")), timeout=1, project_active=lambda: trusted
    )


def test_without_a_hub_it_says_how_to_turn_formatting_on() -> None:
    assert format_summary(None) == 'formatting is off (set "format": true in settings)'


def test_a_hub_with_nothing_listed_says_so() -> None:
    assert format_summary(hub()) == "No formatters found."


def test_each_formatter_is_one_line_with_its_state_and_extensions_and_warnings_follow() -> None:
    a = FormatterSpec("ruff\x1b", ("/bin/ruff",), frozenset({".py", ".pyi"}), {}, "global")
    b = FormatterSpec("mine", ("/bin/m",), frozenset({".m"}), {}, "project")
    text = format_summary(hub(a, b, missing=("gofmt",), warnings=("format: odd",), trusted=False))
    assert text.splitlines() == [
        "ruff\\x1b (global) found  .py .pyi",
        "mine (project) off: untrusted  .m",
        "gofmt (builtin) not found",
        "format: odd",
    ]
