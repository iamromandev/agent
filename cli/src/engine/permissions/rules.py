"""One permission rule: how its text is parsed and which tool calls it matches. Pure, no I/O."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

# These tools run their `command` in the user's shell (`bash`, and `term_start` through `$SHELL -c`): a rule for
# them has to read a command line, not just a string.
SHELL_TOOLS = frozenset({"bash", "term_start"})

# What a command may consist of to match an allow rule that has a prefix: plain words and arguments only (the prefix
# the user wrote is held to it too, so a rule cannot be made to allow a `;`). The shell is
# whatever `$SHELL` is (bash, zsh, fish), and each has its own ways to run code from inside an argument
# (`=(cmd)`, `*(e:cmd:)`, `(cmd)`, `{a,b}`, `$(cmd)`, quotes, redirects, chains). An allow-list instead of a list of
# bad operators: anything not named here is a command nobody approved, so the dialog decides.
PLAIN_COMMAND = re.compile(r"[A-Za-z0-9 _./:=@%+,-]*")

# Where a deny rule looks for the start of a command inside a command line: after any of these.
_COMMAND_START = re.compile(r"[;&|`(){}\n]")

Source = Literal["global", "project"]  # which settings file a rule came from

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_SERVER = re.compile(r"mcp__[A-Za-z0-9_-]+__\*")


class RuleError(ValueError):
    """A rule's text is not valid; the message is the reason."""


@dataclass(frozen=True)
class Rule:
    text: str  # as written, e.g. "bash(git status*)"
    tool: str  # "bash", or "mcp__gh__" with server_wildcard=True
    value: str | None  # None = bare tool
    prefix: bool  # a trailing * inside (...)
    server_wildcard: bool  # "mcp__<server>__*"

    def matches(self, tool: str, args: Mapping[str, Any], key: str | None) -> bool:
        """Does a call of `tool` with `args` match, so that an allow rule may approve it? `key` is the tool's key
        argument (`Tool.grant_arg`). As narrow as it can be: when in doubt, no."""
        if self.server_wildcard:
            return tool.startswith(self.tool)
        if tool != self.tool:
            return False
        if self.value is None:
            return True
        actual = _argument(tool, args, key)
        if actual is None:
            return False
        if not self.prefix:
            return actual == self.value
        if not actual.startswith(self.value):
            return False
        return tool not in SHELL_TOOLS or PLAIN_COMMAND.fullmatch(actual) is not None

    def denies(self, tool: str, args: Mapping[str, Any], key: str | None) -> bool:
        """Does a call match, so that a deny rule must refuse it? As wide as it can be: for a shell command the rule
        is looked for at the start of every command in the line (after `;`, `&&`, a newline, `$(`, ...) and after
        leading whitespace. Best effort: `env rm`, `sudo rm` or `/bin/rm` do not start with `rm`."""
        if self.server_wildcard:
            return tool.startswith(self.tool)
        if tool != self.tool:
            return False
        if self.value is None:
            return True
        actual = _argument(tool, args, key)
        if actual is None:
            return False
        if tool not in SHELL_TOOLS:
            return actual == self.value if not self.prefix else actual.startswith(self.value)
        starts = [part.lstrip() for part in (actual, *_COMMAND_START.split(actual))]  # the whole line, then each command
        if self.prefix:
            return any(command.startswith(self.value) for command in starts)
        return any(command.rstrip() == self.value for command in starts) or actual == self.value


def _argument(tool: str, args: Mapping[str, Any], key: str | None) -> str | None:
    """The text a valued rule is compared with: the call's key argument, if the tool has one and it is a string."""
    key = resolve_key(tool, key)
    if key is None:
        return None
    actual = args.get(key)
    return actual if isinstance(actual, str) else None


def resolve_key(tool: str, key: str | None) -> str | None:
    """The argument a rule or grant for `tool` is tied to: its key argument, else `command` for bash (the same
    rule as `Grants`), else none."""
    return key or ("command" if tool == "bash" else None)


def parse_rule(text: str) -> Rule:
    """The rule for `text`, or a `RuleError` saying what is wrong with it."""
    if not text.strip():
        raise RuleError("empty rule")
    if "(" not in text:
        return _bare(text)
    name, _, rest = text.partition("(")
    if not _NAME.fullmatch(name):
        raise RuleError("invalid tool name")
    if not rest.endswith(")"):
        raise RuleError("unbalanced parentheses")
    value = rest[:-1]
    if not value:
        raise RuleError("empty value")
    prefix = value.endswith("*")
    if prefix:
        value = value[:-1]
    if "*" in value:
        raise RuleError("* only at the end")
    return Rule(text, name, value, prefix, False)


def _bare(text: str) -> Rule:
    if _SERVER.fullmatch(text):
        return Rule(text, text[:-1], None, False, True)
    if "*" in text:
        raise RuleError("* only in the mcp__<server>__* form")
    if ")" in text:
        raise RuleError("unbalanced parentheses" if _NAME.fullmatch(text.split(")")[0]) else "invalid tool name")
    if not _NAME.fullmatch(text):
        raise RuleError("invalid tool name")
    return Rule(text, text, None, False, False)


def exact_rule_text(tool: str, value: str | None) -> str:
    """The rule that allows exactly this call: `tool(value)`, or the bare `tool` when it has no key argument.

    Raises `RuleError` when no rule says exactly that (a `*` in the value would make it a prefix or invalid).
    """
    text = tool if value is None else f"{tool}({value})"
    rule = parse_rule(text)
    if value is not None and (rule.prefix or rule.value != value):
        raise RuleError("the value ends in *, which a rule would read as a prefix")
    return text


def name_matcher(text: str) -> Callable[[str], bool]:
    """A test on a tool name for `text`: an exact name, `<prefix>*` (names that start with it), `mcp__<server>__*`, or
    a lone `*` for every tool. No argument value: `tool(value)` is a `RuleError`, like any other invalid text."""
    if text == "*":
        return lambda tool: True
    if text.endswith("*") and text != "*" and "(" not in text and _NAME.fullmatch(text[:-1]):
        prefix = text[:-1]
        return lambda tool: tool.startswith(prefix)
    rule = parse_rule(text)
    if rule.value is not None:
        raise RuleError("a matcher names tools, not arguments")
    return lambda tool: rule.matches(tool, {}, None)
