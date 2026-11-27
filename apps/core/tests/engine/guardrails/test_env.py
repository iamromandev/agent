"""The child environment (R8-R10): secret-looking variables are removed unless kept on purpose."""

from __future__ import annotations

from core.engine.guardrails.env import EnvFilter

SECRET_NAMES = [
    "GITHUB_TOKEN",
    "NPM_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_WHATEVER",
    "MY_API_KEY",
    "STRIPE_KEY",
    "SOME_KEY_ID",
    "OPENAI_APIKEY",
    "DB_PASSWORD",
    "DB_PASSWD",
    "GOOGLE_CREDENTIALS",
    "client_secret",
    "github_token",
]


def test_secret_looking_names_are_removed() -> None:
    for name in SECRET_NAMES:
        assert EnvFilter(frozenset()).filter({name: "value", "PATH": "/bin"}) == {"PATH": "/bin"}, name


def test_other_names_are_kept() -> None:
    for name in ["HOME", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "KEYBOARD", "KEYCHAIN", "SSH_AUTH_SOCK", "TERM"]:
        assert EnvFilter(frozenset()).filter({name: "value"}) == {name: "value"}, name


def test_ac9_the_example_environment() -> None:
    env = {"GITHUB_TOKEN": "ghp_x", "MY_API_KEY": "k", "AWS_REGION": "eu-west-1", "PATH": "/bin"}

    assert EnvFilter(frozenset()).filter(env) == {"AWS_REGION": "eu-west-1", "PATH": "/bin"}


def test_keep_env_keeps_a_name_whatever_its_case() -> None:
    env = {"GITHUB_TOKEN": "ghp_x", "npm_token": "n", "OTHER_TOKEN": "o"}

    assert EnvFilter(frozenset({"GITHUB_TOKEN", "NPM_TOKEN"})).filter(env) == {"GITHUB_TOKEN": "ghp_x", "npm_token": "n"}


def test_filter_does_not_change_its_input() -> None:
    env = {"GITHUB_TOKEN": "ghp_x"}

    EnvFilter(frozenset()).filter(env)

    assert env == {"GITHUB_TOKEN": "ghp_x"}


def test_removed_lists_the_names_sorted_and_never_a_value() -> None:
    env = {"ZED_TOKEN": "secret-z", "PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "secret-a", "AWS_REGION": "r"}

    assert EnvFilter(frozenset()).removed(env) == ["AWS_SECRET_ACCESS_KEY", "ZED_TOKEN"]


def test_removed_leaves_out_kept_names() -> None:
    assert EnvFilter(frozenset({"ZED_TOKEN"})).removed({"ZED_TOKEN": "v", "A_TOKEN": "v"}) == ["A_TOKEN"]


def test_secret_values_are_those_of_removed_names_with_eight_or_more_characters() -> None:
    env = {"A_TOKEN": "12345678", "B_TOKEN": "1234567", "PATH": "/usr/local/bin", "C_SECRET": ""}

    assert EnvFilter(frozenset()).secret_values(env) == {"A_TOKEN": "12345678"}


def test_keep_env_accepts_a_trailing_star_for_a_prefix() -> None:
    env = {"AWS_SECRET_ACCESS_KEY": "s", "aws_session_token": "t", "AWS_REGION": "r", "GITHUB_TOKEN": "g"}
    keep = EnvFilter(frozenset({"AWS_*"}))

    assert keep.filter(env) == {"AWS_SECRET_ACCESS_KEY": "s", "aws_session_token": "t", "AWS_REGION": "r"}
    assert keep.removed(env) == ["GITHUB_TOKEN"]


def test_a_star_that_is_not_trailing_is_not_a_pattern() -> None:
    assert EnvFilter(frozenset({"A*_TOKEN"})).filter({"AB_TOKEN": "v"}) == {}
