"""`guardrails` in the settings files: what each file may set, and what is ignored with a warning (R1, R2)."""

from __future__ import annotations

from pathlib import Path

from src.engine.guardrails.config import GuardConfig, read_guardrails

GLOBAL = Path("/cfg/settings.json")
PROJECT = Path("/ws/.agent/settings.json")


def read(global_data: dict | None, project_data: dict | None) -> tuple[GuardConfig, list[str]]:
    return read_guardrails(global_data, project_data, global_path=GLOBAL, project_path=PROJECT)


def test_without_settings_the_defaults_apply_and_nothing_is_said() -> None:
    assert read(None, None) == (GuardConfig(), [])
    assert read({}, {"permissions": {}}) == (GuardConfig(), [])


def test_the_global_file_sets_everything() -> None:
    config, warnings = read(
        {
            "guardrails": {
                "sandbox": {"enabled": False, "network": False, "write": ["~/extra/cache"]},
                "protected": ["~/secrets/**", "*.key"],
                "keepEnv": ["npm_token", "Other"],
            }
        },
        None,
    )

    assert config == GuardConfig(
        sandbox_enabled=False,
        network=False,
        extra_write=("~/extra/cache",),
        extra_protected=("~/secrets/**", "*.key"),
        keep_env=frozenset({"NPM_TOKEN", "OTHER"}),
    )
    assert warnings == []


def test_the_project_file_may_add_protected_patterns_and_turn_the_network_off() -> None:
    config, warnings = read(
        {"guardrails": {"protected": ["a/**"]}},
        {"guardrails": {"protected": ["b/**"], "sandbox": {"network": False}}},
    )

    assert config.extra_protected == ("a/**", "b/**")
    assert config.network is False
    assert warnings == []


def test_ac1_project_keys_that_loosen_are_ignored_with_a_warning_each() -> None:
    config, warnings = read(
        {"guardrails": {"sandbox": {"network": False}}},
        {"guardrails": {"sandbox": {"enabled": False}, "keepEnv": ["X"]}},
    )

    assert config.sandbox_enabled is True
    assert config.network is False
    assert config.keep_env == frozenset()
    assert len(warnings) == 2
    assert f"{PROJECT}: guardrails.sandbox.enabled can only be set in the global settings; ignored" in warnings
    assert f"{PROJECT}: guardrails.keepEnv can only be set in the global settings; ignored" in warnings


def test_a_project_network_true_and_write_list_are_ignored_with_a_warning() -> None:
    config, warnings = read(
        {"guardrails": {"sandbox": {"network": False}}},
        {"guardrails": {"sandbox": {"network": True, "write": ["~/x"]}}},
    )

    assert config.network is False
    assert config.extra_write == ()
    assert warnings == [
        f"{PROJECT}: guardrails.sandbox.network can only be set in the global settings; ignored",
        f"{PROJECT}: guardrails.sandbox.write can only be set in the global settings; ignored",
    ]


def test_a_project_enabled_true_changes_nothing_and_says_nothing() -> None:
    config, warnings = read(None, {"guardrails": {"sandbox": {"enabled": True}, "keepEnv": []}})

    assert config == GuardConfig()
    assert warnings == []


def test_ac2_a_protected_value_that_is_not_a_list_warns_and_keeps_the_defaults() -> None:
    config, warnings = read({"guardrails": {"protected": 5}}, None)

    assert config.extra_protected == ()
    assert warnings == [f'{GLOBAL}: "guardrails.protected" must be a list of non-empty strings; ignored']


def test_a_list_with_a_non_string_or_empty_entry_is_ignored_as_a_whole() -> None:
    config, warnings = read({"guardrails": {"protected": ["ok", 1], "keepEnv": ["A", ""]}}, None)

    assert config.extra_protected == ()
    assert config.keep_env == frozenset()
    assert len(warnings) == 2


def test_wrong_types_fall_back_to_the_default_per_key() -> None:
    config, warnings = read(
        {"guardrails": {"sandbox": {"enabled": "yes", "network": 0, "write": "x"}, "protected": ["p"]}}, None
    )

    assert config == GuardConfig(extra_protected=("p",))
    assert len(warnings) == 3


def test_guardrails_and_sandbox_must_be_objects() -> None:
    _, first = read({"guardrails": []}, None)
    config, second = read({"guardrails": {"sandbox": 1, "protected": ["p"]}}, None)

    assert first == [f'{GLOBAL}: "guardrails" must be an object; ignored']
    assert second == [f'{GLOBAL}: "guardrails.sandbox" must be an object; ignored']
    assert config.extra_protected == ("p",)


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


def test_a_pattern_in_a_warning_is_escaped() -> None:
    _, warnings = read({"guardrails": {"protected": ["a/../\x1b[31mb"]}}, None)

    assert len(warnings) == 1
    assert "\x1b" not in warnings[0]


def test_a_protected_pattern_is_trimmed() -> None:
    config, warnings = read({"guardrails": {"protected": ["  ~/secrets/** ", "\t*.token\n"]}}, None)

    assert config.extra_protected == ("~/secrets/**", "*.token")
    assert warnings == []


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


def test_a_protected_pattern_ending_in_double_star_does_not_warn() -> None:
    _, warnings = read({"guardrails": {"protected": ["~/secrets/**", "a/*.txt"]}}, None)

    assert warnings == []
