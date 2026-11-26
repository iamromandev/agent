"""The prompt of a headless run: the argument and stdin in one text, built-ins refused, custom commands expanded."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import BinaryIO

from src.engine.commands.loader import Commands
from src.headless import STDIN_LIMIT, UsageError

_SLASH = re.compile(r"/([A-Za-z0-9_-]+)(?:\s+(.*))?", re.DOTALL)


def build_prompt(
    arg: str | None,
    stdin: BinaryIO,
    stdin_is_tty: bool,
    commands: Commands,
    builtins: Iterable[str],
    attach: Callable[[str], str] | None = None,
) -> str:
    """The prompt for `-p`. Without `arg`, stdin is the prompt (a terminal is a usage error). With `arg` and a
    piped stdin, the input follows after a blank line in a fenced block. `stdin` is read as bytes, at most
    `STDIN_LIMIT`, and must be UTF-8. `attach` (the `@file` expansion) sees only what the user wrote, the argument
    or a stdin that is the prompt, after a custom command is expanded: never the piped input block."""
    if arg is None:
        if stdin_is_tty:
            raise UsageError('give a prompt: agent -p "<prompt>", or pipe it on stdin')
        base, piped = _read(stdin), ""
    else:
        base, piped = arg, ("" if stdin_is_tty else _read(stdin))
    base = base.strip()
    if not base:
        raise UsageError("the prompt is empty")
    base = _expand(base, commands, set(builtins))
    if attach is not None:
        base = attach(base)
    if piped.strip():
        piped = piped.rstrip("\n")
        runs = re.findall(r"`+", piped)
        fence = "`" * max(3, max(map(len, runs), default=0) + 1)
        base = f"{base}\n\n<piped input>\n{fence}\n{piped}\n{fence}"
    return base


def _read(stdin: BinaryIO) -> str:
    try:
        data = stdin.read(STDIN_LIMIT + 1)
    except (OSError, ValueError):  # no usable stdin (closed, or not readable): no input
        return ""
    if len(data) > STDIN_LIMIT:
        raise UsageError("stdin is over the limit of 1 MiB")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UsageError("stdin is not valid UTF-8") from exc


def _expand(text: str, commands: Commands, builtins: set[str]) -> str:
    match = _SLASH.fullmatch(text)
    if match is None:
        return text
    name, rest = match.group(1), match.group(2) or ""
    if name in builtins:
        raise UsageError(f"/{name} is not available in headless mode (it is a command of the interactive app)")
    command = commands.get(name)
    return command.expand(rest).strip() if command is not None else text
