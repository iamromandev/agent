"""Region shapes for a masked edit_image: parsing, bounds and the dialog line (R1, R4, R8, AC7)."""

from __future__ import annotations

import pytest
from core.engine.image_mask import Shape, check_region, describe, parse_region


def rect(x: int = 0, y: int = 0, w: int = 10, h: int = 10) -> dict[str, object]:
    return {"shape": "rect", "x": x, "y": y, "w": w, "h": h}


def test_a_valid_region_parses_into_shapes() -> None:
    got = parse_region([rect(1, 2, 3, 4), {**rect(), "shape": "ellipse"}])
    assert got == [Shape("rect", 1, 2, 3, 4), Shape("ellipse", 0, 0, 10, 10)]


@pytest.mark.parametrize(
    "value",
    [
        "x",
        [],
        [rect()] * 9,
        [rect(w=0)],
        [rect(h=0)],
        [{**rect(), "shape": "star"}],
        [{"shape": "rect", "x": 0, "y": 0, "w": 1}],
        [{**rect(), "x": 1.5}],
        [{**rect(), "x": True}],
        [{**rect(), "x": "1"}],
        [{**rect(), "extra": 1}],
        [3],
        [rect(x=2**40)],
    ],
)
def test_an_invalid_region_is_a_message(value: object) -> None:
    got = parse_region(value)
    assert isinstance(got, str)
    assert got


def test_a_shape_wholly_outside_is_named_by_number() -> None:
    shapes = [Shape("rect", 0, 0, 5, 5), Shape("rect", 100, 0, 5, 5)]
    assert check_region(shapes, 100, 50) == "region shape 2 lies outside the source (100\u00d750)"


def test_shapes_that_touch_the_source_are_fine_even_when_they_overhang() -> None:
    assert check_region([Shape("rect", -5, -5, 10, 10), Shape("ellipse", 95, 45, 50, 50)], 100, 50) is None
    assert check_region([Shape("rect", -10, 0, 10, 10)], 100, 50) is not None


def test_describe_counts_each_kind_and_the_feather() -> None:
    a = Shape("rect", 0, 0, 1, 1)
    b = Shape("ellipse", 0, 0, 1, 1)
    assert describe([a], None, 8) == "region: 1 rect, feather 8 px"
    assert describe([a, a, b], None, 0) == "region: 2 rects, 1 ellipse, feather 0 px"
    assert describe([b, b], None, 64) == "region: 2 ellipses, feather 64 px"
    assert describe(None, "m.png", 8) == "mask: m.png, feather 8 px"
