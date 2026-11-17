"""`build_prompt`: the argument and stdin into one prompt, built-ins refused, custom commands expanded."""

from __future__ import annotations

from io import BytesIO

import pytest
from src.engine.commands.builtins import BUILTINS
from src.engine.commands.loader import Command, Commands
from src.headless import STDIN_LIMIT, UsageError
from src.headless.prompt import build_prompt


def commands(**bodies: str) -> Commands:
    return Commands({n: Command(n, b, None, "project") for n, b in bodies.items()}, ())


def build(arg: str | None, stdin: bytes = b"", tty: bool = False, cmds: Commands | None = None) -> str:
    return build_prompt(arg, BytesIO(stdin), tty, cmds or commands(), BUILTINS)


def test_without_an_argument_the_whole_of_stdin_is_the_prompt() -> None:
    assert build(None, b"abc\ndef\n") == "abc\ndef"


def test_an_argument_with_piped_stdin_appends_it_in_a_fenced_block_after_a_blank_line() -> None:
    assert build("explain", b"data\n") == "explain\n\n<piped input>\n```\ndata\n```"


def test_the_fence_is_longer_than_any_run_of_backticks_in_the_input() -> None:
    text = build("x", b"a ```` b ``` c")

    assert text.splitlines()[3] == "`````" and text.endswith("\n`````")


def test_an_argument_with_empty_stdin_is_just_the_argument() -> None:
    assert build("hello", b"  \n") == "hello"


def test_an_argument_on_a_terminal_never_reads_stdin() -> None:
    class Boom(BytesIO):
        def read(self, *_: object) -> bytes:
            raise AssertionError("stdin was read")

    assert build_prompt("hello", Boom(), True, commands(), BUILTINS) == "hello"


def test_no_argument_and_a_terminal_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="prompt"):
        build(None, tty=True)


@pytest.mark.parametrize("arg,stdin", [(None, b""), (None, b" \n\t"), ("  ", b"")])
def test_an_empty_prompt_is_a_usage_error(arg: str | None, stdin: bytes) -> None:
    with pytest.raises(UsageError, match="empty"):
        build(arg, stdin)


def test_stdin_over_one_mib_is_a_usage_error_but_exactly_one_mib_is_fine() -> None:
    assert len(build(None, b"a" * STDIN_LIMIT)) == STDIN_LIMIT
    with pytest.raises(UsageError, match="1 MiB"):
        build(None, b"a" * (STDIN_LIMIT + 1))
    with pytest.raises(UsageError, match="1 MiB"):
        build("x", b"a" * (STDIN_LIMIT + 1))


def test_stdin_that_is_not_utf8_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="UTF-8"):
        build(None, b"\xff\xfe\x00bad")


def test_a_unreadable_stdin_counts_as_no_input() -> None:
    class Closed:
        def read(self, *_: object) -> bytes:
            raise OSError("no stdin")

    assert build_prompt("hi", Closed(), False, commands(), BUILTINS) == "hi"  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("prompt", ["/compact", "/model qwen", "  /help", "/new\nmore"])
def test_a_built_in_name_is_a_usage_error_that_names_it(prompt: str) -> None:
    name = prompt.split()[0].lstrip("/")
    with pytest.raises(UsageError, match=name):
        build(prompt)


def test_a_built_in_name_from_stdin_is_refused_too() -> None:
    with pytest.raises(UsageError, match="compact"):
        build(None, b"/compact")


def test_a_custom_command_expands_with_its_arguments() -> None:
    assert build("/fix the bug", cmds=commands(fix="Fix: $ARGUMENTS")) == "Fix: the bug"


def test_a_custom_command_then_piped_input() -> None:
    assert build("/fix", b"log", cmds=commands(fix="Fix it")) == "Fix it\n\n<piped input>\n```\nlog\n```"


@pytest.mark.parametrize("prompt", ["/etc/passwd please", "/unknown thing", "/", "look at /compact"])
def test_other_slash_text_is_an_ordinary_prompt(prompt: str) -> None:
    assert build(prompt) == prompt

