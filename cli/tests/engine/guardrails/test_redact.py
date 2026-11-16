"""`Redactor`: secrets in tool output become `[redacted:<kind>]` (R14, R15); ordinary text stays as it is."""

from __future__ import annotations

import pytest
from src.engine.guardrails.redact import TAIL_RUN_MAX, Redactor, drop_partial_tail

NONE = Redactor({})

GITHUB = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7\nabcdef==\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        ("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
        ("gho_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
        ("ghu_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
        ("ghs_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
        ("ghr_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
        ("github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz0123456789", "github-token"),
        ("AKIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("ASIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("xoxb-123456789012-abcdefghij", "slack-token"),
        ("xoxp-123456789012-abcdefghij", "slack-token"),
        ("sk-proj-abcdefghijklmnopqrstuvwx", "openai-key"),
        ("sk-ant-api03-abcdefghijklmnopqrstuvwx", "anthropic-key"),
        (JWT, "jwt"),
        (PEM, "private-key"),
    ],
)
def test_each_listed_kind_is_replaced_as_a_whole(secret: str, kind: str) -> None:
    text, count = NONE.redact(f"before {secret} after")

    assert text == f"before [redacted:{kind}] after"
    assert count == 1


def test_a_pem_block_is_replaced_whole_between_ordinary_lines() -> None:
    text, count = NONE.redact(f"key follows\n{PEM}\nbye")

    assert text == "key follows\n[redacted:private-key]\nbye"
    assert count == 1


def test_an_openssh_and_a_plain_private_key_block_count_too() -> None:
    for label in ("OPENSSH PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY"):
        text, count = NONE.redact(f"-----BEGIN {label}-----\nAAAA\n-----END {label}-----")

        assert (text, count) == ("[redacted:private-key]", 1)


def test_a_pem_block_cut_short_is_replaced_to_the_end_of_the_text() -> None:
    text, count = NONE.redact("a\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\nmore")

    assert text == "a\n[redacted:private-key]"
    assert count == 1


def test_a_public_key_block_is_left_alone() -> None:
    block = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkq\n-----END PUBLIC KEY-----"

    assert NONE.redact(block) == (block, 0)


def test_the_anthropic_key_is_not_also_counted_as_an_openai_key() -> None:
    text, count = NONE.redact("sk-ant-api03-abcdefghijklmnopqrstuvwx")

    assert (text, count) == ("[redacted:anthropic-key]", 1)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "aws_secret_access_key = [redacted:aws-secret-key]"),
        ("aws_secret_access_key=abc123", "aws_secret_access_key=[redacted:aws-secret-key]"),
        ("AWS_SECRET_ACCESS_KEY: abc123xyz", "AWS_SECRET_ACCESS_KEY: [redacted:aws-secret-key]"),
        ('"aws_secret_access_key": "abc123xyz"', '"aws_secret_access_key": "[redacted:aws-secret-key]"'),
        ("password=hunter2", "password=[redacted:password]"),
        ("password: hunter2", "password: [redacted:password]"),
        ("DB_PASSWORD=hunter2 next", "DB_PASSWORD=[redacted:password] next"),
        ('{"password": "hun ter2"}', '{"password": "[redacted:password]"}'),
        ("password = 'hun ter2'", "password = '[redacted:password]'"),
        ('"password":"x1"', '"password":"[redacted:password]"'),
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


def test_the_password_value_stops_at_the_line_end() -> None:
    text, count = NONE.redact("password:\nuser: bob")

    assert (text, count) == ("password:\nuser: bob", 0)


def test_several_secrets_are_all_replaced_and_counted() -> None:
    text, count = NONE.redact(f"a {GITHUB} b {JWT} c {GITHUB}\npassword=x")

    assert text == "a [redacted:github-token] b [redacted:jwt] c [redacted:github-token]\npassword=[redacted:password]"
    assert count == 4


def test_redacting_twice_changes_nothing_more() -> None:
    once, _ = NONE.redact(f"{GITHUB} password=abc {PEM}")

    assert NONE.redact(once) == (once, 0)


def test_no_text_no_change() -> None:
    assert NONE.redact("") == ("", 0)


@pytest.mark.parametrize(
    "text",
    [
        "commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
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
        "def login(password: str, other=1)",
        "check(password=password)",
        "password = getpass()",
        "password: str = Field(...)",
        "DB_PASSWORD=DB_PASSWORD",
        "password=None password=null password=true password=False",
        "password: bytes, password: int, password: string",
        "x = os.environ.get(password=input(prompt))",
    ],
)
def test_ordinary_text_is_not_touched(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


def test_an_env_value_is_replaced_with_its_variable_name() -> None:
    redactor = Redactor({"DEPLOY_TOKEN": "s3cr3t-value-123"})

    text, count = redactor.redact("curl -H 'x: s3cr3t-value-123' and again s3cr3t-value-123")

    assert text == "curl -H 'x: [redacted:env:DEPLOY_TOKEN]' and again [redacted:env:DEPLOY_TOKEN]"
    assert count == 2


def test_a_value_shorter_than_eight_characters_is_not_redacted() -> None:
    redactor = Redactor({"A_TOKEN": "1234567", "B_TOKEN": "12345678"})

    assert redactor.redact("1234567 12345678") == ("1234567 [redacted:env:B_TOKEN]", 1)


def test_the_longest_value_wins_when_one_contains_another() -> None:
    redactor = Redactor({"SHORT_TOKEN": "abcdefgh", "LONG_TOKEN": "abcdefgh-and-more"})

    text, count = redactor.redact("x abcdefgh-and-more y abcdefgh z")

    assert text == "x [redacted:env:LONG_TOKEN] y [redacted:env:SHORT_TOKEN] z"
    assert count == 2


def test_a_value_with_regex_characters_is_matched_literally() -> None:
    redactor = Redactor({"PW_SECRET": "a.b*c+d(e)[f]|g$"})

    assert redactor.redact("x a.b*c+d(e)[f]|g$ y aXb*c+d(e)[f]|g$") == (
        "x [redacted:env:PW_SECRET] y aXb*c+d(e)[f]|g$",
        1,
    )


def test_a_value_that_is_also_a_pattern_is_counted_once() -> None:
    redactor = Redactor({"GITHUB_TOKEN": GITHUB})

    text, count = redactor.redact(f"token {GITHUB}")

    assert text == "token [redacted:github-token]"
    assert count == 1


def test_a_value_inside_a_placeholder_is_not_replaced_again() -> None:
    redactor = Redactor({"X_TOKEN": "redacted", "Y_TOKEN": "yvalue-12345"})

    text, _ = redactor.redact(f"{GITHUB} yvalue-12345")

    assert text == "[redacted:github-token] [redacted:env:Y_TOKEN]"


def test_the_repr_never_shows_a_value() -> None:
    redactor = Redactor({"GITHUB_TOKEN": "super-secret-value-1"})

    assert "super-secret-value-1" not in repr(redactor)
    assert "GITHUB_TOKEN" not in repr(redactor)


def test_the_same_value_under_two_names_uses_one_name_deterministically() -> None:
    redactor = Redactor({"B_TOKEN": "same-value-1234", "A_TOKEN": "same-value-1234"})

    assert redactor.redact("same-value-1234") == ("[redacted:env:A_TOKEN]", 1)


# --- review fixes -------------------------------------------------------------------------------------------------


def test_an_orphan_pem_end_line_takes_the_body_before_it_with_it() -> None:
    text, count = NONE.redact("MIIEowIBAAKCAQEA7\nabcdef==\n-----END RSA PRIVATE KEY-----\nafter")

    assert text == "[redacted:private-key]\nafter"
    assert count == 1


def test_an_orphan_pem_end_line_starts_at_the_preceding_blank_line() -> None:
    text, count = NONE.redact("header\n\nsome text\nMIIEow\n-----END PRIVATE KEY-----\nafter")

    assert text == "header\n\n[redacted:private-key]\nafter"
    assert count == 1


def test_a_whole_block_and_a_later_orphan_end_are_both_replaced() -> None:
    text, count = NONE.redact(f"{PEM}\nok\n\nBBBB\n-----END PRIVATE KEY-----")

    assert text == "[redacted:private-key]\nok\n\n[redacted:private-key]"
    assert count == 2


def test_a_public_key_end_line_alone_is_left_alone() -> None:
    block = "MIIBIjANBgkq\n-----END PUBLIC KEY-----"

    assert NONE.redact(block) == (block, 0)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r'{\"password\": \"hunter2\", \"user\": \"bob\"}', r'{\"password\": \"[redacted:password]\", \"user\": \"bob\"}'),
        (r'{\"password\":\"a b\"}', r'{\"password\":\"[redacted:password]\"}'),
        (r'{"password": "ab\"cd-tail", "user": "bob"}', r'{"password": "[redacted:password]", "user": "bob"}'),
        (r"password = 'it\'s-secret'", r"password = '[redacted:password]'"),
        ('password="unterminated value', 'password="[redacted:password]'),
    ],
)
def test_escaped_json_and_escaped_quotes_do_not_leak_the_tail(source: str, expected: str) -> None:
    text, count = NONE.redact(source)

    assert text == expected
    assert count == 1


def test_an_unsigned_jwt_with_an_empty_signature_is_replaced() -> None:
    text, count = NONE.redact("t eyJhbGciOiJub25lIn0.eyJzdWIiOiIxIn0. end")

    assert (text, count) == ("t [redacted:jwt] end", 1)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("f(password=abc)", "f(password=[redacted:password])"),
        ("[password: abc]", "[password: [redacted:password]]"),
        ("{password: abc}", "{password: [redacted:password]}"),
        ("<password=abc>", "<password=[redacted:password]>"),
        ("password=hunter2 ok", "password=[redacted:password] ok"),
    ],
)
def test_a_bare_password_value_stops_at_closing_brackets(source: str, expected: str) -> None:
    assert NONE.redact(source) == (expected, 1)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("task-sk-abcdefghijklmnopqrstuvwxyz", "task-[redacted:openai-key]"),
        ("x ask-sk-proj-abcdefghijklmnopqrstuvwx y", "x ask-[redacted:openai-key] y"),
        ("xghp_abcdefghijklmnopqrstuvwxyz0123456789", "x[redacted:github-token]"),
        ("kgithub_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz0123456789", "k[redacted:github-token]"),
        ("XAKIAIOSFODNN7EXAMPLE", "X[redacted:aws-access-key]"),
        ("kASIAIOSFODNN7EXAMPLE", "k[redacted:aws-access-key]"),
        ("Zxoxb-123456789012-abcdefghij", "Z[redacted:slack-token]"),
        (f"xx{JWT}", "xx[redacted:jwt]"),
    ],
)
def test_a_distinctive_prefix_in_the_middle_of_a_word_is_still_a_secret(source: str, expected: str) -> None:
    assert NONE.redact(source) == (expected, 1)


@pytest.mark.parametrize("text", ["risk-assessment-for-the-whole-system", "task-abcdefghijklmnopqrstuvwxyz"])
def test_sk_inside_a_word_is_still_no_key(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


def test_an_env_value_with_surrounding_whitespace_is_found_without_it() -> None:
    redactor = Redactor({"A_TOKEN": "  spaced-secret-1  \n"})

    assert redactor.redact("x spaced-secret-1 y") == ("x [redacted:env:A_TOKEN] y", 1)


def test_a_multi_line_env_value_is_found_with_crlf_line_ends() -> None:
    redactor = Redactor({"KEY_FILE_TOKEN": "first-line-secret\nsecond-line-secret"})

    text, count = redactor.redact("a first-line-secret\r\nsecond-line-secret b")

    assert (text, count) == ("a [redacted:env:KEY_FILE_TOKEN] b", 1)


def test_each_line_of_a_multi_line_env_value_is_redacted_on_its_own() -> None:
    redactor = Redactor({"KEY_FILE_TOKEN": "first-line-secret\nshort\nsecond-line-secret"})

    text, count = redactor.redact("only second-line-secret here, short stays")

    assert (text, count) == ("only [redacted:env:KEY_FILE_TOKEN] here, short stays", 1)


def test_an_env_value_inside_json_escapes_is_found() -> None:
    redactor = Redactor({"Q_TOKEN": 'pa"ss\\word\tx-12345'})

    text, count = redactor.redact('{"k": "pa\\"ss\\\\word\\tx-12345"}')

    assert (text, count) == ('{"k": "[redacted:env:Q_TOKEN]"}', 1)


def test_an_env_value_with_non_ascii_is_found_in_its_json_form() -> None:
    redactor = Redactor({"U_TOKEN": "pässword-123456"})

    assert redactor.redact('"p\\u00e4ssword-123456" "pässword-123456"') == (
        '"[redacted:env:U_TOKEN]" "[redacted:env:U_TOKEN]"',
        2,
    )


def test_two_overlapping_env_values_are_covered_as_one() -> None:
    redactor = Redactor({"A_TOKEN": "abcdefgh12345678", "B_TOKEN": "12345678zzzzzzzz"})

    text, count = redactor.redact("x abcdefgh12345678zzzzzzzz y")

    assert text == "x [redacted:env:A_TOKEN] y"
    assert count == 1


def test_overlapping_env_values_are_covered_when_the_later_one_ends_first() -> None:
    redactor = Redactor({"A_TOKEN": "abcdefgh12345678", "B_TOKEN": "12345678zzzzzzzz", "C_TOKEN": "5678zzzzzzzzQQQQ"})

    text, _ = redactor.redact("x abcdefgh12345678zzzzzzzzQQQQ y")

    assert text == "x [redacted:env:A_TOKEN] y"


# --- second review ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        'getpass.getpass("Password: ")',
        "print('Password: ', end='')",
        'raise ValueError("password: " + x)',
        'echo "Password: " && read -s x',
        "prompt = 'Enter password: ' + name",
        'log("password: ", user)',
        'password_prompt = "Password:"',
    ],
)
def test_a_prompt_or_message_that_names_a_password_is_not_a_value(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


def test_a_password_after_a_closed_string_on_the_same_line_is_still_redacted() -> None:
    text, count = NONE.redact("""print('Password: ', end=''); password='hunter2'""")

    assert text == """print('Password: ', end=''); password='[redacted:password]'"""
    assert count == 1


def test_a_password_value_after_a_prompt_string_is_not_hidden_by_it() -> None:
    text, count = NONE.redact("echo 'Password: ' password=hunter2")

    assert text == "echo 'Password: ' password=[redacted:password]"
    assert count == 1


@pytest.mark.parametrize(
    "text",
    [
        '"password": {"a": 1}',
        '"password": ["a"]',
        "password: [a, b]",
        "password=%(pw)s",
        "password=(a, b)",
        "password=<redacted>",
        "password: <your password here>",
    ],
)
def test_a_password_followed_by_a_structure_or_template_is_left_alone(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("password=$ecretpass", "password=[redacted:password]"),
        ("password=$2b$12$abcdefghijklmnopqrstuuABCDEFGHIJKLMNOPQRSTUVWXYZ01234", "password=[redacted:password]"),
        ("password=$DB_PASS_extra", "password=[redacted:password]"),
        ("password=Summer(2024)", "password=[redacted:password])"),
        ("password=Hunter2(Hunter2", "password=[redacted:password]"),
        ("password: Optionalx", "password: [redacted:password]"),
        ("password=Foo[bar", "password=[redacted:password]"),
        ("password=Field(...)", "password=[redacted:password])"),
    ],
)
def test_values_that_only_look_like_references_or_calls_are_redacted(source: str, expected: str) -> None:
    assert NONE.redact(source) == (expected, 1)


@pytest.mark.parametrize(
    "text",
    [
        "password=$DB_PASSWORD",
        "password=${DB_PASSWORD}",
        "password=$HOME/pw",
        "password=$(cat /run/secrets/pw)",
        "password = getpass()",
        "password = os.getenv('PW')",
        "password = input(prompt)",
        "password=get_password(user, host)",
        "password: Optional[str] = None",
        "password: list[str]",
        "password: dict[str, int]",
        "password: Union[str, None]",
        "password: Literal['a']",
        "password: tuple[str, ...]",
        "password: set[str]",
    ],
)
def test_a_shell_reference_a_call_and_a_known_annotation_are_not_values(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    "text",
    ["x.password = y.password", "self.password=args.password", "password=settings.db_password"],
)
def test_an_attribute_reference_is_not_a_value(text: str) -> None:
    assert NONE.redact(text) == (text, 0)


@pytest.mark.parametrize(
    "source",
    [
        r'{\\\"password\\\": \\\"hunter2\\\", \\\"user\\\": \\\"bob\\\"}',
        r'{\"password\": \"ab\\\"cd-tail\", \"user\": \"bob\"}',
        r'{\\\"password\\\": \\\"ab\\\\\\\"cd-tail\\\", \\\"user\\\": \\\"bob\\\"}',
        r'{\"password\": \"a\\\\\\\\\", \"user\": \"bob\"}',
    ],
)
def test_json_escaped_at_any_depth_does_not_leak_the_tail_of_a_password(source: str) -> None:
    text, count = NONE.redact(source)

    assert count == 1
    assert "tail" not in text
    assert "hunter2" not in text
    assert "[redacted:password]" in text
    assert "bob" in text  # the next field is left alone


def test_a_doubly_escaped_password_keeps_the_text_around_it() -> None:
    text, _ = NONE.redact(r'{\\\"password\\\": \\\"ab\\\\\\\"cd\\\", \\\"user\\\": \\\"bob\\\"}')

    assert text == r'{\\\"password\\\": \\\"[redacted:password]\\\", \\\"user\\\": \\\"bob\\\"}'


def test_an_aws_secret_in_doubly_escaped_json_is_redacted() -> None:
    text, count = NONE.redact(r'{\\\"aws_secret_access_key\\\": \\\"abc123xyz\\\"}')

    assert (text, count) == (r'{\\\"aws_secret_access_key\\\": \\\"[redacted:aws-secret-key]\\\"}', 1)


def test_an_orphan_end_line_inside_a_test_assertion_is_left_alone() -> None:
    source = 'assert "-----END PRIVATE KEY-----" in text\nassert "-----END RSA PRIVATE KEY-----" not in other'

    assert NONE.redact(source) == (source, 0)


def test_an_orphan_end_line_after_prose_is_left_alone() -> None:
    source = "The key file ends with this line.\n-----END PRIVATE KEY-----\nthat is all"

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


def test_a_last_run_longer_than_any_token_is_kept() -> None:
    long_line = "x" * (TAIL_RUN_MAX + 1)

    assert drop_partial_tail(f"start {long_line}") == (f"start {long_line}", 0)


def test_an_env_value_is_found_after_a_long_unclosed_placeholder_start() -> None:
    redactor = Redactor({"A_TOKEN": "value-number-0001-secret"})

    text, count = redactor.redact("[redacted:" + "x" * 300 + " value-number-0001-secret")

    assert text == "[redacted:" + "x" * 300 + " [redacted:env:A_TOKEN]"
    assert count == 1
