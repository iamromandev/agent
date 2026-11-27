"""Templates for workflow arguments: `{name.path}` placeholders and `$map`. Plain text substitution, nothing evaluated."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PLACEHOLDER = re.compile(r"[a-z_][a-z0-9_]*(?:\.[A-Za-z0-9_]+)*")
LOCAL = frozenset({"item", "index"})
MAP_KEYS = frozenset({"$map", "to"})


class TemplateError(Exception):
    """A template that cannot be rendered (a missing path, a stray brace)."""


@dataclass(frozen=True)
class Scope:
    values: dict[str, Any]


def _pieces(text: str) -> list[str | tuple[str]]:
    """`text` as literal strings and `(expression,)` placeholders. `{{` and `}}` are literal braces."""
    out: list[str | tuple[str]] = []
    literal: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "{":
            if text.startswith("{{", i):
                literal.append("{")
                i += 2
                continue
            end = text.find("}", i + 1)
            if end < 0:
                raise TemplateError(f"unclosed '{{' in {text[:60]!r}")
            expr = text[i + 1 : end]
            if not PLACEHOLDER.fullmatch(expr):
                raise TemplateError(f"bad placeholder {{{expr}}}")
            if literal:
                out.append("".join(literal))
                literal = []
            out.append((expr,))
            i = end + 1
        elif char == "}":
            if not text.startswith("}}", i):
                raise TemplateError(f"stray '}}' in {text[:60]!r}")
            literal.append("}")
            i += 2
        else:
            start = i
            while i < n and text[i] not in "{}":
                i += 1
            literal.append(text[start:i])
    if literal:
        out.append("".join(literal))
    return out


def _lookup(expr: str, scope: Scope) -> Any:
    root, *path = expr.split(".")
    if root not in scope.values:
        raise TemplateError(f"no value for {{{expr}}}")
    value = scope.values[root]
    for part in path:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise TemplateError(f"no value for {{{expr}}}")
    return value


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _render_string(text: str, scope: Scope) -> Any:
    pieces = _pieces(text)
    if len(pieces) == 1 and isinstance(pieces[0], tuple):
        return _lookup(pieces[0][0], scope)  # exactly one placeholder: the value keeps its type
    return "".join(p if isinstance(p, str) else _text(_lookup(p[0], scope)) for p in pieces)


def render_text(text: str, scope: Scope) -> str:
    """`text` with every placeholder replaced by its text form (JSON for a list or object), even when the string
    is exactly one placeholder: for prompts, which are always text. Raises `TemplateError`."""
    return "".join(p if isinstance(p, str) else _text(_lookup(p[0], scope)) for p in _pieces(text))


def _is_map(value: dict[str, Any]) -> bool:
    return "$map" in value


def _map_parts(value: dict[str, Any]) -> tuple[str, Any]:
    if set(value) != MAP_KEYS:
        raise TemplateError("$map needs exactly the keys '$map' and 'to'")
    over = value["$map"]
    if not isinstance(over, str) or not PLACEHOLDER.fullmatch(over):
        raise TemplateError("$map must be a path like 'step.list'")
    return over, value["to"]


def render(value: Any, scope: Scope) -> Any:
    """`value` with every placeholder replaced. A string that is exactly one placeholder gives the value itself,
    any other string text; what is substituted is never expanded again. Raises `TemplateError`."""
    if isinstance(value, str):
        return _render_string(value, scope)
    if isinstance(value, list):
        return [render(item, scope) for item in value]
    if isinstance(value, dict):
        if _is_map(value):
            over, to = _map_parts(value)
            items = _lookup(over, scope)
            if not isinstance(items, list):
                raise TemplateError(f"$map over {over}: not a list")
            return [render(to, Scope({**scope.values, "item": item, "index": n})) for n, item in enumerate(items, 1)]
        return {key: render(item, scope) for key, item in value.items()}
    return value


def placeholders(value: Any) -> set[str]:
    """The root names a template uses (`item` and `index` too, except inside a `$map`'s `to`, which binds them).
    Raises `TemplateError` for a template that cannot be parsed."""
    if isinstance(value, str):
        return {p[0].split(".")[0] for p in _pieces(value) if isinstance(p, tuple)}
    if isinstance(value, list):
        return set().union(*(placeholders(item) for item in value)) if value else set()
    if isinstance(value, dict):
        if _is_map(value):
            over, to = _map_parts(value)
            return {over.split(".")[0]} | (placeholders(to) - LOCAL)
        return set().union(*(placeholders(item) for item in value.values())) if value else set()
    return set()
