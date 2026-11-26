"""A small JSON Schema subset: enough to check what a model step answers. Nothing is evaluated or fetched."""

from __future__ import annotations

import math
from typing import Any

MAX_DEPTH = 16
TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
    }
)


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_ok(expected: str, value: object) -> bool:
    kind = _kind(value)
    if expected == "number":
        return kind in ("integer", "number")
    return kind == expected


def _same(a: object, b: object) -> bool:
    """JSON equality: a bool equals only a bool, numbers compare by value, containers element by element."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | dict) or isinstance(b, list | dict):
        return False
    return a == b


def _non_finite(value: object) -> bool:
    """Whether a NaN or an infinity is anywhere in `value` (iterative: no recursion limit)."""
    todo = [value]
    while todo:
        item = todo.pop()
        if isinstance(item, float) and not math.isfinite(item):
            return True
        if isinstance(item, list):
            todo.extend(item)
        elif isinstance(item, dict):
            todo.extend(item.values())
    return False


def validate(schema: dict[str, Any], value: object, path: str = "$") -> str | None:
    """The first way `value` breaks `schema` as `<path>: <problem>`, or None. `schema` passed `check_schema`.
    NaN and infinities are never valid (they are not JSON)."""
    if path == "$" and _non_finite(value):
        return f"{path}: NaN and infinity are not allowed"
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_ok(expected, value):
        return f"{path}: expected {expected}, got {_kind(value)}"
    if "enum" in schema and not any(_same(value, allowed) for allowed in schema["enum"]):
        return f"{path}: not one of the allowed values"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}: missing required key '{key}'"
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                if (error := validate(properties[key], item, f"{path}.{key}")) is not None:
                    return error
            elif schema.get("additionalProperties", True) is False:
                return f"{path}: unexpected key '{key}'"
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{path}: needs at least {schema['minItems']} items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path}: needs at most {schema['maxItems']} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for n, item in enumerate(value):
                if (error := validate(item_schema, item, f"{path}[{n}]")) is not None:
                    return error
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path}: needs at least {schema['minLength']} characters"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path}: needs at most {schema['maxLength']} characters"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: must be at least {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: must be at most {schema['maximum']}"
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check(schema: object, path: str, depth: int) -> str | None:
    if depth > MAX_DEPTH:
        return f"{path}: schema nested too deep (limit {MAX_DEPTH})"
    if not isinstance(schema, dict):
        return f"{path}: a schema must be an object"
    for key in schema:
        if key not in KEYWORDS:
            return f"{path}: unsupported keyword '{key}'"
    kind = schema.get("type")
    if "type" in schema and (not isinstance(kind, str) or kind not in TYPES):
        return f"{path}: unsupported type '{kind}'"
    if "required" in schema and not (
        isinstance(schema["required"], list) and all(isinstance(k, str) for k in schema["required"])
    ):
        return f"{path}: required must be a list of names"
    if "enum" in schema and not isinstance(schema["enum"], list):
        return f"{path}: enum must be a list"
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        return f"{path}: additionalProperties must be true or false"
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            return f"{path}: {key} must be a whole number"
    for key in ("minimum", "maximum"):
        if key in schema and not _is_number(schema[key]):
            return f"{path}: {key} must be a number"
    properties = schema.get("properties")
    if "properties" in schema:
        if not isinstance(properties, dict):
            return f"{path}: properties must be an object"
        for name, sub in properties.items():
            if (error := _check(sub, f"{path}.{name}", depth + 1)) is not None:
                return error
    if "items" in schema and (error := _check(schema["items"], f"{path}.items", depth + 1)) is not None:
        return error
    return None


def check_schema(schema: object) -> str | None:
    """Why `schema` cannot be used (an unsupported keyword or type, a bad value, nesting over 16), or None."""
    return _check(schema, "$", 1)
