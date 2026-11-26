"""`guardrails` in the settings files: what each file may set, and what is ignored with a warning (R1, R2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.guardrails.config import GuardConfig, read_guardrails

GLOBAL = Path("/cfg/settings.json")
PROJECT = Path("/ws/.agent/settings.json")


def read(global_data: dict | None, project_data: dict | None) -> tuple[GuardConfig, list[str]]:
    return read_guardrails(global_data, project_data, global_path=GLOBAL, project_path=PROJECT)


UNSUPPORTED = "guardrails.sandbox is no longer supported and is ignored"
SANDBOX_SHAPES = [{"enabled": False, "network": False, "write": ["~/x"]}, "x", {"enabled": [1]}, 1, None, []]


def test_the_global_file_sets_everything() -> None:
    config, warnings = read(
        {"guardrails": {"protected": ["~/secrets/**", "*.key"], "keepEnv": ["npm_token", "Other"]}},
        None,
    )

    assert config == GuardConfig(
        extra_protected=("~/secrets/**", "*.key"),
        keep_env=frozenset({"NPM_TOKEN", "OTHER"}),
    )
    assert warnings == []


def test_the_project_file_may_add_protected_patterns() -> None:
    config, warnings = read({"guardrails": {"protected": ["a/**"]}}, {"guardrails": {"protected": ["b/**"]}})

    assert config.extra_protected == ("a/**", "b/**")
    assert warnings == []


@pytest.mark.parametrize("shape", SANDBOX_SHAPES)
def test_a_sandbox_key_in_each_file_gives_one_warning_and_other_keys_still_apply(shape: object) -> None:
    config, warnings = read(
        {"guardrails": {"sandbox": shape, "protected": ["g/**"], "keepEnv": ["tok"]}},
        {"guardrails": {"sandbox": shape, "protected": ["p/**"]}},
    )

    assert warnings == [f"{GLOBAL}: {UNSUPPORTED}", f"{PROJECT}: {UNSUPPORTED}"]
    assert config == GuardConfig(extra_protected=("g/**", "p/**"), keep_env=frozenset({"TOK"}))


def test_a_file_whose_only_guardrails_key_is_sandbox_gives_one_warning_and_nothing_else() -> None:
    config, warnings = read({"guardrails": {"sandbox": {"enabled": True}}}, None)

    assert config == GuardConfig()
    assert warnings == [f"{GLOBAL}: {UNSUPPORTED}"]


def test_ac2_a_protected_value_that_is_not_a_list_warns_and_keeps_the_defaults() -> None:
    config, warnings = read({"guardrails": {"protected": 5}}, None)

    assert config.extra_protected == ()
    assert warnings == [f'{GLOBAL}: "guardrails.protected" must be a list of non-empty strings; ignored']


def test_a_list_with_a_non_string_or_empty_entry_is_ignored_as_a_whole() -> None:
    config, warnings = read({"guardrails": {"protected": ["ok", 1], "keepEnv": ["A", ""]}}, None)

    assert config.extra_protected == ()
    assert config.keep_env == frozenset()
    assert len(warnings) == 2


def test_guardrails_must_be_an_object() -> None:
    _, first = read({"guardrails": []}, None)

    assert first == [f'{GLOBAL}: "guardrails" must be an object; ignored']


def test_a_bad_project_value_warns_with_the_project_path() -> None:
    config, warnings = read(None, {"guardrails": {"protected": "nope"}})

    assert config == GuardConfig()
    assert warnings == [f'{PROJECT}: "guardrails.protected" must be a list of non-empty strings; ignored']


def test_text_from_a_file_is_escaped_in_warnings() -> None:
    evil = Path("/ws/\x1b[31m/settings.json")

    _, warnings = read_guardrails(None, {"guardrails": {"keepEnv": ["X"]}}, project_path=evil)

    assert warnings
    assert all(w.isascii() and w.isprintable() for w in warnings)
    assert "\\x1b" in warnings[0]


def test_a_protected_pattern_with_a_dotdot_segment_is_ignored_with_a_warning() -> None:
    config, warnings = read(
        {"guardrails": {"protected": ["~/a/../b", "ok/**"]}},
        {"guardrails": {"protected": ["../x/**", "fine"]}},
    )

    assert config.extra_protected == ("ok/**", "fine")
    assert len(warnings) == 2
    assert warnings[0].startswith("/cfg/settings.json: ") and '"~/a/../b"' in warnings[0] and '".."' in warnings[0]
    assert warnings[1].startswith("/ws/.agent/settings.json: ") and '"../x/**"' in warnings[1]


def test_a_bare_tilde_pattern_is_ignored_with_a_warning_that_suggests_the_folder_form() -> None:
    config, warnings = read({"guardrails": {"protected": ["~", "ok"]}}, None)

    assert config.extra_protected == ("ok",)
    assert len(warnings) == 1
    assert "~/**" in warnings[0]


def test_a_protected_pattern_that_is_empty_after_trimming_is_ignored_with_a_warning() -> None:
    from src.engine.guardrails.config import _Reader

    warnings: list[str] = []

    kept = _Reader("/cfg", warnings).patterns(("  ", "ok"))

    assert kept == ("ok",)
    assert len(warnings) == 1
    assert "empty" in warnings[0] and warnings[0].startswith("/cfg: ")


def test_a_protected_pattern_ending_in_slash_star_is_kept_with_a_warning_that_suggests_double_star() -> None:
    config, warnings = read({"guardrails": {"protected": ["~/secrets/*"]}}, {"guardrails": {"protected": ["vault/*"]}})

    assert config.extra_protected == ("~/secrets/*", "vault/*")
    assert len(warnings) == 2
    assert "direct children" in warnings[0] and '"~/secrets/**"' in warnings[0]
    assert warnings[1].startswith("/ws/.agent/settings.json: ") and '"vault/**"' in warnings[1]
