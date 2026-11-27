from __future__ import annotations

from typing import Any

from core.engine.workflows.estimates import DEFAULT_RATE, OTHER_RATE, estimate_plan, unit
from core.engine.workflows.model import PlannedCall
from core.engine.workflows.timings import TimingStore


def pc(tool: str, args: dict[str, Any] | None = None, key: str = "k") -> PlannedCall:
    return PlannedCall(key, key, None, tool, args or {}, "d")


def reel(scenes: int) -> list[PlannedCall]:
    img = {"width": 576, "height": 1024}
    return [
        *(pc("generate_image", img, f"pic/{i}") for i in range(scenes)),
        *(pc("speak", {}, f"voice/{i}") for i in range(scenes)),
        pc("assemble_video", {"scenes": [{}] * scenes}, "video"),
    ]


def test_image_units_are_pixels_times_steps() -> None:
    assert unit("generate_image", {"width": 512, "height": 256, "steps": 3}) == ("generate_image", 512 * 256 * 3)
    assert unit("generate_image", {}) == ("generate_image", 1024 * 1024 * 9)
    assert unit("edit_image", {}) == ("edit_image", 1024 * 1024 * 4)
    assert unit("edit_image", {"width": 512, "height": 512, "steps": 2}) == ("edit_image", 512 * 512 * 2)


def test_video_unit_is_the_scene_count_and_others_are_per_call() -> None:
    assert unit("assemble_video", {"scenes": [{}, {}, {}]}) == ("assemble_video", 3)
    assert unit("assemble_video", {}) == ("assemble_video", 1)
    assert unit("speak", {"text": "x"}) == ("speak", 1)


def test_default_reel_is_within_half_of_the_measured_730_seconds() -> None:
    seconds, source = estimate_plan(reel(3), TimingStore(None))

    assert 365 <= seconds <= 1095
    assert source == "a rough guess"


def test_unknown_tool_uses_the_other_rate() -> None:
    assert estimate_plan([pc("write")], TimingStore(None))[0] == round(OTHER_RATE)


def test_records_replace_defaults_with_their_median() -> None:
    store = TimingStore(None)
    for value in (1e-4, 2e-4, 9e-4):
        store.add("generate_image", value)
    calls = [pc("generate_image", {"width": 1000, "height": 1000, "steps": 10})]

    seconds, source = estimate_plan(calls, store)

    assert seconds == round(2e-4 * 1000 * 1000 * 10)
    assert source == "learned from 3 calls"


def test_a_mix_says_partly_learned() -> None:
    store = TimingStore(None)
    store.add("speak", 8.0)
    calls = [pc("speak"), pc("speak"), pc("assemble_video", {"scenes": [{}]})]

    seconds, source = estimate_plan(calls, store)

    assert seconds == round(8.0 * 2 + DEFAULT_RATE["assemble_video"])
    assert source == "partly learned (2 of 3 calls)"


def test_a_learned_reel_is_within_half_of_its_recorded_total() -> None:
    store = TimingStore(None)
    img_rate = 310.0 / (576 * 1024 * 9)
    for _ in range(3):
        store.add("generate_image", img_rate)
        store.add("speak", 5.0)
        store.add("assemble_video", 25.0 / 3)
    real_total = 3 * 310 + 3 * 5 + 25

    seconds, source = estimate_plan(reel(3), store)

    assert abs(seconds - real_total) <= real_total * 0.5
    assert source.startswith("learned from")


# --- upscale_image is per output pixel ---------------------------------------------------------


def test_upscale_unit_is_the_output_pixels_of_the_source_when_it_is_known() -> None:
    seen: list[dict[str, Any]] = []

    def pixels(args: dict[str, Any]) -> int | None:
        seen.append(args)
        return 1024 * 1024

    assert unit("upscale_image", {"scale": 2}, pixels) == ("upscale_image", float(1024 * 1024))
    assert seen == [{"scale": 2}]


def test_upscale_unit_assumes_a_1024_source_times_scale_squared_without_a_source() -> None:
    assert unit("upscale_image", {}) == ("upscale_image", float(2048 * 2048))
    assert unit("upscale_image", {"scale": 3}) == ("upscale_image", float(3072 * 3072))
    assert unit("upscale_image", {"scale": 3}, lambda args: None) == ("upscale_image", float(3072 * 3072))
    assert unit("upscale_image", {"scale": True}) == ("upscale_image", float(2048 * 2048))


def test_default_upscale_rate_matches_the_two_measurements() -> None:
    big = estimate_plan([pc("upscale_image", {"scale": 2})], TimingStore(None))[0]
    small = estimate_plan([pc("upscale_image", {"scale": 2})], TimingStore(None), lambda args: 1024 * 1024)[0]

    assert abs(big - 257) <= 5  # 1024 -> 2048
    assert 30 <= small <= 120  # 512 -> 1024 took 60 s
