"""`Redactor`: secrets in tool output become `[redacted:<kind>]` (R14, R15); ordinary text stays as it is."""

from __future__ import annotations

import pytest
from src.engine.guardrails.redact import Redactor, drop_partial_tail

NONE = Redactor({})

GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7\nabcdef==\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        (PEM, "private-key"),
    ],
)
def test_each_listed_kind_is_replaced_as_a_whole(secret: str, kind: str) -> None:
    text, count = NONE.redact(f"before {secret} after")

    assert text == f"before [redacted:{kind}] after"
    assert count == 1


def test_an_openssh_and_a_plain_private_key_block_count_too() -> None:
    for label in ("OPENSSH PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY"):
        text, count = NONE.redact(f"-----BEGIN {label}-----\nAAAA\n-----END {label}-----")

        assert (text, count) == ("[redacted:private-key]", 1)


def test_a_pem_block_cut_short_is_replaced_to_the_end_of_the_text() -> None:
    text, count = NONE.redact("a\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\nmore")

    assert text == "a\n[redacted:private-key]"
    assert count == 1


def test_the_anthropic_key_is_not_also_counted_as_an_openai_key() -> None:
    text, count = NONE.redact("sk-ant-api03-abcdefghijklmnopqrstuvwx")

    assert (text, count) == ("[redacted:anthropic-key]", 1)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_access_key = [redacted:aws-secret-key]"),
        ("AWS_SECRET_ACCESS_KEY: abc123xyz", "AWS_SECRET_ACCESS_KEY: [redacted:aws-secret-key]"),
        ('"aws_secret_access_key": "abc123xyz"', '"aws_secret_access_key": "[redacted:aws-secret-key]"'),
        ("password=hunter2", "password=[redacted:password]"),
        ('{"password": "hun ter2"}', '{"password": "[redacted:password]"}'),
        ("Password: hunter2,", "Password: [redacted:password],"),
    ],
)
def test_an_assignment_loses_only_its_value(source: str, expected: str) -> None:
    text, count = NONE.redact(source)

    assert text == expected
    assert count == 1


def test_an_empty_password_and_an_already_redacted_one_are_not_counted() -> None:
    for source in ('password=""', "password=", "password: [redacted:env:X]", "password is required"):
        assert NONE.redact(source) == (source, 0)


def test_several_secrets_are_all_replaced_and_counted() -> None:
    text, count = NONE.redact(f"a {GITHUB} b {JWT} c {GITHUB}\npassword=x")

    assert text == "a [redacted:github-token] b [redacted:jwt] c [redacted:github-token]\npassword=[redacted:password]"
    assert count == 4


ORDINARY = [
    "commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904",
    "id 123e4567-e89b-12d3-a456-426614174000",
    "sk-short",
    "sk-" + "a" * 19,
    "task-abcdefghijklmnopqrstuvwxyz",
    "risk-assessment-for-the-whole-system",
    "SGVsbG8gd29ybGQsIHRoaXMgaXMganVzdCBiYXNlNjQgdGV4dA==",
    "eyJhbGciOiJIUzI1NiJ9",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0",
    "ghp_tooshort",
    "gh_abcdefghijklmnopqrstuvwxyz0123456789",
    "AKIA1234",
    "AKIAIOSFODNN7EXAMPLEXTRA",
    "xoxb-short",
    "the password policy requires 12 characters",
    "def login(user, password):",
    "password = getpass()",
    "DB_PASSWORD=DB_PASSWORD",
    "password=None password=null password=true password=False",
]


def test_ordinary_text_is_not_touched() -> None:
    for text in ORDINARY:
        assert NONE.redact(text) == (text, 0), text


def test_a_value_that_is_also_a_pattern_is_counted_once() -> None:
    redactor = Redactor({"GITHUB_TOKEN": GITHUB})

    text, count = redactor.redact(f"token {GITHUB}")

    assert text == "token [redacted:github-token]"
    assert count == 1


# --- review fixes -------------------------------------------------------------------------------------------------


def test_an_orphan_pem_end_line_starts_at_the_preceding_blank_line() -> None:
    text, count = NONE.redact("header\n\nsome text\nMIIEow\n-----END PRIVATE KEY-----\nafter")

    assert text == "header\n\n[redacted:private-key]\nafter"
    assert count == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('password="unterminated value', 'password="[redacted:password]'),
    ],
)
def test_escaped_json_and_escaped_quotes_do_not_leak_the_tail(source: str, expected: str) -> None:
    text, count = NONE.redact(source)

    assert text == expected
    assert count == 1


def test_a_multi_line_env_value_is_found_with_crlf_line_ends() -> None:
    redactor = Redactor({"KEY_FILE_TOKEN": "first-line-secret\nsecond-line-secret"})

    text, count = redactor.redact("a first-line-secret\r\nsecond-line-secret b")

    assert (text, count) == ("a [redacted:env:KEY_FILE_TOKEN] b", 1)


def test_two_overlapping_env_values_are_covered_as_one() -> None:
    redactor = Redactor({"A_TOKEN": "abcdefgh12345678", "B_TOKEN": "12345678zzzzzzzz"})

    text, count = redactor.redact("x abcdefgh12345678zzzzzzzz y")

    assert text == "x [redacted:env:A_TOKEN] y"
    assert count == 1


# --- second review ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "password=<redacted>",
    ],
)
def test_a_password_followed_by_a_structure_or_template_is_left_alone(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    "text",
    [
        "password=${DB_PASSWORD}",
    ],
)
def test_a_shell_reference_a_call_and_a_known_annotation_are_not_values(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    "text",
    ["x.password = y.password"],
)
def test_an_attribute_reference_is_not_a_value(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    "source",
    [
        r'{\\\"password\\\": \\\"ab\\\\\\\"cd-tail\\\", \\\"user\\\": \\\"bob\\\"}',
    ],
)
def test_json_escaped_at_any_depth_does_not_leak_the_tail_of_a_password(source: str) -> None:
    text, count = NONE.redact(source)

    assert count == 1
    assert "tail" not in text
    assert "hunter2" not in text
    assert "[redacted:password]" in text
    assert "bob" in text  # the next field is left alone


def test_an_orphan_end_line_inside_a_test_assertion_is_left_alone() -> None:
    source = 'assert "-----END PRIVATE KEY-----" in text\nassert "-----END RSA PRIVATE KEY-----" not in other'

    assert NONE.redact(source) == (source, 0)


def test_an_orphan_end_line_after_a_long_base64_line_is_redacted() -> None:
    body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7"
    text, count = NONE.redact(f"{body}\n-----END PRIVATE KEY-----\nafter")

    assert (text, count) == ("[redacted:private-key]\nafter", 1)


def test_the_partial_last_word_of_a_cut_text_is_dropped_and_counted() -> None:
    assert drop_partial_tail("a line\nghp_abcdefghijkl") == ("a line\n", 16)
    assert drop_partial_tail("ends with a space ") == ("ends with a space ", 0)
    assert drop_partial_tail("oneword") == ("", 7)
    assert drop_partial_tail("") == ("", 0)
