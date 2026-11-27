"""`build_prompt`: the argument and stdin into one prompt, built-ins refused, custom commands expanded."""

from __future__ import annotations

from io import BytesIO

import pytest
from core.engine.commands.builtins import BUILTINS
from core.engine.commands.loader import Command, Commands
from src.headless import STDIN_LIMIT, UsageError
from src.headless.prompt import build_prompt


def commands(**bodies: str) -> Commands:
    return Commands({n: Command(n, b, None, "project") for n, b in bodies.items()}, ())


def build(arg: str | None, stdin: bytes = b"", tty: bool = False, cmds: Commands | None = None) -> str:
    return build_prompt(arg, BytesIO(stdin), tty, cmds or commands(), BUILTINS)


def test_no_argument_and_a_terminal_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="prompt"):
        build(None, tty=True)


def test_stdin_over_one_mib_is_a_usage_error_but_exactly_one_mib_is_fine() -> None:
    assert len(build(None, b"a" * STDIN_LIMIT)) == STDIN_LIMIT
    with pytest.raises(UsageError, match="1 MiB"):
        build(None, b"a" * (STDIN_LIMIT + 1))
    with pytest.raises(UsageError, match="1 MiB"):
        build("x", b"a" * (STDIN_LIMIT + 1))


def test_stdin_that_is_not_utf8_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="UTF-8"):
        build(None, b"\xff\xfe\x00bad")


@pytest.mark.parametrize("prompt", ["/new\nmore"])
def test_a_built_in_name_is_a_usage_error_that_names_it(prompt: str) -> None:
    name = prompt.split()[0].lstrip("/")
    with pytest.raises(UsageError, match=name):
        build(prompt)


def test_a_custom_command_expands_with_its_arguments() -> None:
    assert build("/fix the bug", cmds=commands(fix="Fix: $ARGUMENTS")) == "Fix: the bug"
