from __future__ import annotations

from typing import Any

from src.engine.workflows.schema import check_schema, validate

SCENES: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 5}},
                "required": ["text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenes"],
}


def test_valid_value_passes() -> None:
    assert validate(SCENES, {"scenes": [{"text": "hi"}]}) is None


def test_wrong_type_names_the_path() -> None:
    assert validate(SCENES, {"scenes": "x"}) == "$.scenes: expected array, got string"


def test_missing_required_key() -> None:
    assert validate(SCENES, {}) == "$: missing required key 'scenes'"


def test_additional_properties_false_rejects_extra_key() -> None:
    assert validate(SCENES, {"scenes": [{"text": "a", "x": 1}]}) == "$.scenes[0]: unexpected key 'x'"


def test_additional_properties_default_allows_extra_key() -> None:
    assert validate({"type": "object"}, {"x": 1}) is None


def test_array_bounds() -> None:
    assert validate(SCENES, {"scenes": []}) == "$.scenes: needs at least 1 items"
    too_many = {"scenes": [{"text": "a"}] * 4}
    assert validate(SCENES, too_many) == "$.scenes: needs at most 3 items"


def test_string_length_bounds() -> None:
    assert validate(SCENES, {"scenes": [{"text": ""}]}) == "$.scenes[0].text: needs at least 1 characters"
    assert validate(SCENES, {"scenes": [{"text": "toolong"}]}) == "$.scenes[0].text: needs at most 5 characters"


def test_integer_and_number_reject_bool() -> None:
    assert validate({"type": "integer"}, True) is not None
    assert validate({"type": "number"}, False) is not None
    assert validate({"type": "integer"}, 3) is None
    assert validate({"type": "number"}, 3) is None
    assert validate({"type": "number"}, 3.5) is None
    assert validate({"type": "integer"}, 3.5) is not None


def test_null_and_boolean() -> None:
    assert validate({"type": "null"}, None) is None
    assert validate({"type": "boolean"}, True) is None
    assert validate({"type": "boolean"}, 1) is not None


def test_enum_and_bounds() -> None:
    assert validate({"enum": ["a", "b"]}, "c") == "$: not one of the allowed values"
    assert validate({"enum": ["a", "b"]}, "a") is None
    assert validate({"type": "integer", "minimum": 2}, 1) == "$: must be at least 2"
    assert validate({"type": "integer", "maximum": 2}, 3) == "$: must be at most 2"


def test_check_schema_accepts_a_supported_schema() -> None:
    assert check_schema(SCENES) is None


def test_check_schema_refuses_an_unknown_keyword() -> None:
    assert check_schema({"type": "object", "pattern": "x"}) == "$: unsupported keyword 'pattern'"


def test_check_schema_refuses_unknown_nested_keyword() -> None:
    assert check_schema({"type": "array", "items": {"$ref": "#"}}) == "$.items: unsupported keyword '$ref'"


def test_check_schema_refuses_a_bad_type() -> None:
    assert check_schema({"type": "date"}) == "$: unsupported type 'date'"


def test_check_schema_refuses_a_non_object() -> None:
    assert check_schema([1]) == "$: a schema must be an object"


def test_check_schema_caps_the_depth() -> None:
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(20):
        schema = {"type": "array", "items": schema}
    error = check_schema(schema)
    assert error is not None
    assert "too deep" in error


def test_enum_does_not_treat_a_bool_as_a_number() -> None:
    assert validate({"enum": [1]}, True) is not None
    assert validate({"enum": [0]}, False) is not None
    assert validate({"enum": [True]}, 1) is not None
    assert validate({"enum": [True]}, True) is None


def test_enum_numbers_compare_by_value() -> None:
    assert validate({"enum": [1]}, 1.0) is None
    assert validate({"enum": [1.5]}, 1.5) is None
    assert validate({"enum": [1]}, 2) is not None


def test_enum_is_type_aware_inside_lists_and_objects() -> None:
    assert validate({"enum": [[1]]}, [True]) is not None
    assert validate({"enum": [{"a": 1}]}, {"a": True}) is not None
    assert validate({"enum": [{"a": 1}]}, {"a": 1}) is None


def test_non_finite_numbers_are_refused() -> None:
    schema = {"type": "number", "maximum": 5}
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert validate(schema, bad) is not None
    assert validate({"type": "object"}, {"a": [float("nan")]}) is not None
