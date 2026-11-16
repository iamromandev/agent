"""Redaction of secrets in tool output (R14, R15): what the model, the transcript and the session file get.

Deliberately narrow (the spec prefers missing a secret to masking ordinary output): every pattern needs a fixed
prefix or key name, and short or unmarked strings (git SHAs, UUIDs, `sk-` plus a few letters) are left alone.
Text only; nothing on disk changes. Values of environment variables are never part of a repr or an error text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from src.engine.guardrails.env import MIN_SECRET_LENGTH

PLACEHOLDER = "[redacted:"  # every placeholder starts with this: `[redacted:<kind>]`, `[redacted:env:<NAME>]`
# A quote of a key, or one escaped any number of times: JSON inside a JSON string has `\"`, one more level `\\\"`.
_QUOTE_OPTION = r"\\*+[\"']?"
_ASSIGN = rf"{_QUOTE_OPTION}[ \t]*[=:][ \t]*"
_NOT_DONE = r"(?!\[redacted:)"  # a value that is already a placeholder stays one
# A placeholder as the env pass skips it. The name is bounded: with `*`, a text of `[redacted:` without a `]` read to its
# end from each one (13 s for 200 KB). Kinds and variable names are far shorter than the bound.
_PLACEHOLDER_SKIP = r"\[redacted:[^\]\n]{0,128}\]"
_BASE64URL = "A-Za-z0-9_-"
_IDENT_LOOKBACK = 64  # how far back the key name of an assignment is read

# A bare password value that is not a secret: a type, a literal, the key's own name, a call, a shell reference.
_NOT_SECRETS = frozenset(
    {"str", "string", "none", "null", "true", "false", "int", "integer", "bytes", "bool", "boolean", "float"}
    | {"password", "secretstr", "any", "object", "optional"}
)
# What a value that is no secret looks like. Each is narrow: a password that only resembles one (`Summer(2024)`,
# `$ecretpass`, a bcrypt hash `$2b$12$...`) is still redacted.
_ANNOTATION = re.compile(r"(?:Optional|Union|Literal|list|dict|tuple|set)\[")  # `Optional[str`
_CALL = re.compile(r"[a-z_][a-z0-9_.]*\((?:[^()\n]{0,80}\))?")  # `getpass(`, `os.getenv('X')`, `input(prompt)`
_SHELL_REF = re.compile(r"\$(?:\(|\{?[A-Z_][A-Z0-9_]*(?![A-Za-z0-9_]))")  # `$(cmd`, `$DB_PASSWORD`, `${DB_PASSWORD`
_BRACED_REF = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*")  # `${password}`: braces make a reference of any name
_DOTTED = re.compile(r"[\w.]+")

# A placeholder as this module writes it: `[redacted:<kind>]` (kinds are lower case) or `[redacted:env:<NAME>]`.
_PLACEHOLDER_TEXT = re.compile(r"\[redacted:(?:[a-z-]+|env:[A-Za-z0-9_]+)\]")

Scan = Callable[[str, list[int]], str]  # (text, counter) -> text; the counter gets one per replacement


@dataclass(frozen=True)
class _Pass:
    hint: str  # lower case text every match contains: a cheap test before running the scan
    scan: Scan


def adds_placeholder(old: str, new: str) -> bool:
    """Whether `new` holds an exact placeholder more often than `old` does (`write` over a file, `edit`'s two strings).

    A model that copies a result back into a file writes the placeholder over the real value, which is what
    this refuses; a file that documents placeholders, or an edit that leaves them where they are, is no concern."""
    if PLACEHOLDER not in new:
        return False
    found = Counter(_PLACEHOLDER_TEXT.findall(new))
    if not found:
        return False
    had = Counter(_PLACEHOLDER_TEXT.findall(old)) if PLACEHOLDER in old else Counter()
    return any(count > had[text] for text, count in found.items())


def _placeholder(kind: str) -> str:
    return f"[redacted:{kind}]"


def _regex_pass(
    kind: str,
    regex: str,
    hint: str,
    groups: tuple[str, ...] = (),
    flags: int = 0,
) -> _Pass:
    """A pass that replaces what `regex` matches (the part in the first matching group of `groups`, when given).

    Every regex here must run in linear time on any text (tests in `test_perf.py`): no nested or overlapping
    unbounded repeats, and a start that cannot match fails at once."""
    compiled = re.compile(regex, flags)
    placeholder = _placeholder(kind)

    def replace(match: re.Match[str], counter: list[int]) -> str:
        counter[0] += 1
        if not groups:
            return placeholder
        group = next(name for name in groups if match.start(name) != -1)
        start = match.start()
        return match.string[start : match.start(group)] + placeholder + match.string[match.end(group) : match.end()]

    def scan(text: str, counter: list[int]) -> str:
        return compiled.sub(lambda match: replace(match, counter), text)

    return _Pass(hint, scan)


def _jwt_pass() -> _Pass:
    """`eyJ` header, payload and signature (empty for an unsigned token), found without a regex over the header.

    A regex that starts at every `eyJ` rescans the whole run of base64url characters from each one (`eyJeyJeyJ...` is
    quadratic), and a lookbehind that rules those starts out would also drop a token that follows a letter. Here the
    run is measured once: when no `.` follows it, no `eyJ` inside it can start a token either."""
    run = re.compile(f"[{_BASE64URL}]+")  # the header after `eyJ`
    rest = re.compile(rf"\.[{_BASE64URL}]+\.[{_BASE64URL}]*")
    placeholder = _placeholder("jwt")

    def scan(text: str, counter: list[int]) -> str:
        out: list[str] = []
        done = 0  # text before this is in `out`
        position = 0
        while (start := text.find("eyJ", position)) != -1:
            header = run.match(text, start + 3)
            tail = rest.match(text, header.end()) if header is not None else None
            if header is None or tail is None:
                position = header.end() if header is not None else start + 3
                continue
            out.append(text[done:start])
            out.append(placeholder)
            counter[0] += 1
            done = position = tail.end()
        if not out:
            return text
        out.append(text[done:])
        return "".join(out)

    return _Pass("eyj", scan)


_END_LINE = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")
_BASE64_LINE = re.compile(r"[A-Za-z0-9+/=]+")
_LINE_NUMBER = re.compile(r" {0,8}\d{1,9}\t")
_BASE64_MIN = 4  # a line of key material is at least this long (the last line of a body can be short: `abcdef==`)


def _without_line_number(line: str) -> str:
    """`line` without the number the `read` tool puts before each line (`     4<tab>`), and without blanks around it."""
    numbered = _LINE_NUMBER.match(line)
    return (line[numbered.end() :] if numbered else line).strip()


def _ends_a_base64_line(text: str, at: int, floor: int) -> bool:
    """Whether the END marker at `at` starts its line and the line before it is base64: prose, a test assertion that
    quotes the marker or a log line is not key material, and what comes before it is not touched."""
    newline = text.rfind("\n", floor, at)
    if newline == -1 or _without_line_number(text[newline + 1 : at]):
        return False
    previous_end = newline - 1 if newline > floor and text[newline - 1] == "\r" else newline
    previous_start = text.rfind("\n", floor, previous_end) + 1
    line = _without_line_number(text[previous_start:previous_end])
    return len(line) >= _BASE64_MIN and _BASE64_LINE.fullmatch(line) is not None


def _orphan_end_pass() -> _Pass:
    """An END line whose BEGIN line is not in the text (a tool window or `term_read(lines=N)` started inside the
    block): everything from the start of the text, or the last blank line, up to it is the key body. Runs after the
    whole-block pass, so every END line left is one without a BEGIN. Only an END line that starts its line after a
    line of base64 counts, so a sentence or an assertion that quotes the marker is left as it is."""
    placeholder = _placeholder("private-key")

    def scan(text: str, counter: list[int]) -> str:
        out: list[str] = []
        done = 0
        for end_line in _END_LINE.finditer(text):
            at = end_line.start()
            if not _ends_a_base64_line(text, at, done):
                continue
            start = done
            blank = text.rfind("\n\n", done, at)
            if blank != -1:
                start = blank + 2
            blank = text.rfind("\r\n\r\n", done, at)
            if blank != -1:
                start = max(start, blank + 4)
            out.append(text[done:start])
            out.append(placeholder)
            counter[0] += 1
            done = end_line.end()
        if not out:
            return text
        out.append(text[done:])
        return "".join(out)

    return _Pass("private key-----", scan)


def _password_is_no_secret(text: str, key: int, bare: re.Match[str]) -> bool:
    """A bare value that is a type, a literal, a template, a reference or a call: not a secret. `key` is where the word
    `password` starts."""
    value = bare.group()
    lowered = value.lower()
    if all(part in _NOT_SECRETS for part in lowered.split("|")):
        return True
    if value[0] in "{[(<" or value.startswith("%("):  # a structure, a placeholder `<redacted>`, `%(pw)s`
        return True
    if _SHELL_REF.match(value) or _ANNOTATION.match(value):
        return True
    if _BRACED_REF.fullmatch(value) and text.startswith("}", bare.end()):
        return True
    call = _CALL.match(text, bare.start())
    if call is not None and call.end() >= bare.end():
        return True
    if "." in value and _DOTTED.fullmatch(value) and "password" in value.rpartition(".")[2].lower():
        return True  # `x.password = y.password`
    first = key
    while first > 0 and key - first < _IDENT_LOOKBACK and (text[first - 1].isalnum() or text[first - 1] == "_"):
        first -= 1
    return lowered == text[first : key + len("password")].lower()  # `check(password=password)`


class _QuoteParity:
    """Whether the number of unescaped quotes before a position on its line is odd. Asked for increasing positions, so
    the text is counted once (a count from the start of the line for every key would be quadratic)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._at = 0
        self._counts = {'"': 0, "'": 0}

    def odd(self, quote: str, at: int) -> bool:
        text, start = self._text, self._at
        if at < start:
            start, self._counts = 0, {'"': 0, "'": 0}
        newline = text.rfind("\n", start, at)
        if newline != -1:
            start, self._counts = newline + 1, {'"': 0, "'": 0}
        for mark in self._counts:
            self._counts[mark] += text.count(mark, start, at) - text.count("\\" + mark, start, at)
        self._at = at
        return self._counts[quote] % 2 == 1


_PASSWORD_HEAD = re.compile(rf"password(?P<key_quote>{_QUOTE_OPTION})[ \t]*[=:][ \t]*", re.IGNORECASE)
_OPENING = re.compile(r"(\\*+)([\"']?)")  # at the start of a value: backslashes, then a quote when there is one
# A bare value is read up to `_BARE_CHECK` characters for the checks that can reject it, and to its end only when it
# is a secret: a text of many keys without a space in between must not read to its end from each of them.
_BARE_CHECK = 256
_BARE = re.compile(rf"[^\s\"',;)\]}}>]{{1,{_BARE_CHECK}}}")
_BARE_REST = re.compile(r"[^\s\"',;)\]}>]*+")
_QUOTE_RUN = {
    '"': re.compile(r'(?<!\\)(\\*+)"|\n'),
    "'": re.compile(r"(?<!\\)(\\*+)'|\n"),
}


def _closing_quote(text: str, start: int, quote: str, escapes: int) -> int:
    """Where a quoted value that starts at `start` ends: its closing quote, the end of the line, or the end of the
    text. The quote is written with `escapes` backslashes (JSON inside JSON strings: 0, 1, 3, 7...). Inside the value
    a run of backslashes before a quote either escapes the quote or is the closing run plus pairs of backslashes that
    are one backslash of the value, so a run closes the value when it is `escapes` plus a multiple of
    `2 * (escapes + 1)` long; with no escapes that is an even run, the usual rule."""
    period = 2 * (escapes + 1)
    for match in _QUOTE_RUN[quote].finditer(text, start):
        run = match.group(1)
        if run is None:  # a line end
            return match.start()
        if len(run) >= escapes and (len(run) - escapes) % period == 0:
            return match.start()
    return len(text)


def _password_value(text: str, head: re.Match[str], parity: _QuoteParity) -> tuple[int, int] | None:
    """The span of the secret after a `password` key, or None when this one is not a value."""
    position = head.end()
    opening = _OPENING.match(text, position)
    assert opening is not None  # it can match the empty string
    backslashes, quote = opening.groups()
    if quote == "'" and backslashes:
        return None
    if not quote:  # a bare value (backslashes that precede no quote belong to it)
        if text.startswith(PLACEHOLDER, position):
            return None
        bare = _BARE.match(text, position)
        if bare is None or _password_is_no_secret(text, head.start(), bare):
            return None
        rest = _BARE_REST.match(text, bare.end())
        assert rest is not None  # it can match the empty string
        return position, rest.end()
    start = opening.end()
    if text.startswith(PLACEHOLDER, start):
        return None
    # `print('Password: ', end='')`: the quote after an unquoted key closes a string that opened before the key.
    if not head.group("key_quote") and not backslashes and parity.odd(quote, head.start()):
        return None
    end = _closing_quote(text, start, quote, len(backslashes))
    return (start, end) if end > start else None


def _password_pass() -> _Pass:
    """The value of a `password` assignment, replaced; the rest of the line stays.

    Done in code, not in one regex, so that the checks that need context (is the quote after the key the end of a
    string that started before it; does the value look like a reference) run before the value is read, and a rejected
    key costs nothing more: the search goes on right after it. A quoted value runs to its closing quote (see
    `_closing_quote`: any depth of JSON escaping), or to the end of the line when there is none."""
    placeholder = _placeholder("password")

    def scan(text: str, counter: list[int]) -> str:
        out: list[str] = []
        done = 0
        position = 0
        parity = _QuoteParity(text)
        while (head := _PASSWORD_HEAD.search(text, position)) is not None:
            position = head.end()
            span = _password_value(text, head, parity)
            if span is None:
                continue
            start, end = span
            out.append(text[done:start])
            out.append(placeholder)
            counter[0] += 1
            done = position = end
        if not out:
            return text
        out.append(text[done:])
        return "".join(out)

    return _Pass("password", scan)


# Order matters: the PEM block goes first (its body may hold anything), `sk-ant-` before the general `sk-`.
# Only `sk-` keeps a check on what comes before it (`task-<20 letters>` is no key): a lookbehind after the literal,
# so the search still skips ahead to the next `sk-`, and a rejected one never hides a real key behind it
# (`task-sk-<key>`). The other prefixes are distinctive: a token right after a letter is still a token.
_PASSES: tuple[_Pass, ...] = (
    # A block without its END line (output cut short) is redacted to the end of the text: fail closed.
    _regex_pass(
        "private-key",
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
        "private key-----",
        flags=re.DOTALL,
    ),
    _orphan_end_pass(),
    _regex_pass("github-token", r"github_pat_[A-Za-z0-9_]{22,}", "github_pat_"),
    _regex_pass("github-token", r"gh[pousr]_[A-Za-z0-9]{36,}", "gh"),
    _regex_pass("aws-access-key", r"(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])", ""),
    _regex_pass("slack-token", r"xox[abprs]-[A-Za-z0-9-]{10,}", "xox"),
    _regex_pass("anthropic-key", r"sk-ant-(?<![A-Za-z0-9]sk-ant-)[A-Za-z0-9_-]{20,}", "sk-ant-"),
    _regex_pass("openai-key", r"sk-(?<![A-Za-z0-9]sk-)[A-Za-z0-9_-]{20,}", "sk-"),
    _jwt_pass(),
    _regex_pass(
        "aws-secret-key",
        rf"aws_secret_access_key{_ASSIGN}(?:\\*+[\"'])?{_NOT_DONE}(?P<v>[^\s\"',;\\)\]}}>]+)",
        "aws_secret_access_key",
        ("v",),
        re.IGNORECASE,
    ),
    _password_pass(),
)


TAIL_RUN_MAX = 4_096  # the longest last word `drop_partial_tail` removes: JWTs and keys fit, a minified file does not


def drop_partial_tail(text: str) -> tuple[str, int]:
    """`(text without its last run of non-whitespace characters, how many characters that was)`.

    For text that was cut at a window: a secret that straddles the cut is not recognised from its first characters
    alone (`ghp_` and 20 more is under the length of a token) and, once the rest of the text has shrunk by redaction,
    would stay as the last thing in the output. The last word of a cut text is cut anyway. A run longer than
    `TAIL_RUN_MAX` is no token a pattern knows (a minified file, a long line of one character) and stays."""
    cut = max(text.rfind(space) for space in (" ", "\n", "\t", "\r")) + 1
    if len(text) - cut > TAIL_RUN_MAX:
        return text, 0
    return text[:cut], len(text) - cut


def _variants(value: str) -> set[str]:
    """The forms of a secret that can show up in output: as it is, trimmed, with CRLF line ends, as one line of a
    multi-line value, and inside a JSON string. Only those of at least `MIN_SECRET_LENGTH` characters."""
    forms = {value, value.strip()}
    if "\n" in value or "\r" in value:
        forms.add(value.replace("\r\n", "\n").replace("\n", "\r\n"))
        forms.update(line.strip() for line in value.splitlines())
    for form in tuple(forms):
        forms.add(json.dumps(form)[1:-1])
        forms.add(json.dumps(form, ensure_ascii=False)[1:-1])
    return {form for form in forms if len(form) >= MIN_SECRET_LENGTH}


@dataclass(frozen=True)
class Redactor:
    # name -> value of the variables left out of child environments. Never in the repr: it holds secrets.
    env_values: Mapping[str, str] = field(repr=False)
    _env_regex: re.Pattern[str] | None = field(init=False, repr=False, compare=False, default=None)
    _env_names: Mapping[str, str] = field(init=False, repr=False, compare=False, default_factory=dict)
    _env_first: frozenset[str] = field(init=False, repr=False, compare=False, default=frozenset())

    def __post_init__(self) -> None:
        # One alternation, longest form first: a secret that contains another one is replaced as a whole, and the
        # placeholders of this pass are never searched again. Equal lengths: by name, so the result is stable.
        usable = sorted(
            ((name, value) for name, value in self.env_values.items() if len(value) >= MIN_SECRET_LENGTH),
            key=lambda item: (-len(item[1]), item[0]),
        )
        names: dict[str, str] = {}
        for name, value in usable:
            for form in sorted(_variants(value)):
                names.setdefault(form, name)  # the same text under two names: the first by the order above
        if names:
            # Placeholders of the pattern pass are matched first only to be skipped: a value like "redacted" must not
            # be found inside them.
            ordered = sorted(names, key=lambda form: (-len(form), form))
            regex = re.compile("|".join([_PLACEHOLDER_SKIP, *(re.escape(form) for form in ordered)]))
            object.__setattr__(self, "_env_regex", regex)
            object.__setattr__(self, "_env_names", names)
            object.__setattr__(self, "_env_first", frozenset(form[0] for form in names))

    def redact(self, text: str) -> tuple[str, int]:
        """`(new text, number of replacements)`."""
        counter = [0]
        lowered = text.lower()
        for scan in _PASSES:
            if scan.hint not in lowered:
                continue
            changed = scan.scan(text, counter)
            if changed != text:
                text, lowered = changed, changed.lower()
        if self._env_regex is not None:
            text = self._redact_env(text, self._env_regex, counter)
        return text, counter[0]

    def _redact_env(self, text: str, regex: re.Pattern[str], counter: list[int]) -> str:
        """Every stretch covered by a value, as one span: values that overlap (`abcd1234` and `1234wxyz` in
        `abcd1234wxyz`) leave no tail, and a placeholder of the pattern pass is never touched."""
        names, first = self._env_names, self._env_first
        out: list[str] = []
        done = 0
        position = 0
        while (match := regex.search(text, position)) is not None:
            start, end = match.span()
            name = names.get(match.group())
            if name is None:  # a placeholder: keep it
                position = end
                continue
            inside = start + 1
            while inside < end:  # a value that starts inside this one and ends after it extends the span
                if text[inside] in first:
                    other = regex.match(text, inside)
                    if other is not None and other.end() > end and other.group() in names:
                        end = other.end()
                inside += 1
            counter[0] += 1
            out.append(text[done:start])
            out.append(_placeholder(f"env:{name}"))
            done = position = end
        if not out:
            return text
        out.append(text[done:])
        return "".join(out)
