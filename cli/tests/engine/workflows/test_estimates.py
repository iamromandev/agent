from __future__ import annotations

from src.engine.workflows.estimates import estimate


def test_generate_image_scales_with_pixels_and_steps() -> None:
    assert estimate("generate_image", {"width": 1024, "height": 1024, "steps": 9}) == round(1024 * 1024 * 9 * 4e-7)
    assert estimate("generate_image", {"width": 512, "height": 512, "steps": 9}) == round(512 * 512 * 9 * 4e-7)


def test_generate_image_defaults() -> None:
    assert estimate("generate_image", {}) == round(1024 * 1024 * 9 * 4e-7)
    assert estimate("generate_image", {"steps": 4}) == round(1024 * 1024 * 4 * 4e-7)


def test_fixed_estimates() -> None:
    assert estimate("speak", {}) == 10
    assert estimate("assemble_video", {}) == 60
    assert estimate("upscale_image", {}) == 60
    assert estimate("edit_image", {}) == 300


def test_anything_else_is_thirty() -> None:
    assert estimate("write", {}) == 30
