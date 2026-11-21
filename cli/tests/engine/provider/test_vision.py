"""VisionCheck: one answer per model, a failure counts as no vision and warns once."""

from __future__ import annotations

import asyncio

from src.engine.provider.vision import VisionCheck

from tests.fakes import FakeCatalog


async def test_a_model_with_the_vision_capability_has_vision() -> None:
    catalog = FakeCatalog(["v", "t"], capabilities={"v": ("completion", "vision")})
    check = VisionCheck(catalog)
    assert await check.has("v") is True
    assert await check.has("t") is False


async def test_the_answer_is_asked_once_per_model() -> None:
    catalog = FakeCatalog(["v"], capabilities={"v": ("vision",)})
    check = VisionCheck(catalog)
    for _ in range(3):
        await check.has("v")
    assert catalog.capability_calls == ["v"]


async def test_a_failure_is_no_vision_with_one_warning_per_model() -> None:
    catalog = FakeCatalog(["a", "b"], fail=True)
    check = VisionCheck(catalog)
    warnings: list[str] = []
    assert await check.has("a", warnings.append) is False
    assert await check.has("a", warnings.append) is False
    assert len(warnings) == 1
    assert warnings[0].startswith("could not read the capabilities of 'a' (")
    assert warnings[0].endswith("); images are not sent")
    await check.has("b", warnings.append)
    assert len(warnings) == 2
    assert catalog.capability_calls == ["a", "a", "b"]  # a failure is not remembered: the next message asks again


async def test_a_failure_is_retried_and_a_later_answer_is_kept() -> None:
    catalog = FakeCatalog(["v"], fail=True, capabilities={"v": ("vision",)})
    check = VisionCheck(catalog)
    warnings: list[str] = []
    assert await check.has("v", warnings.append) is False
    catalog.fail = False  # Ollama is back
    assert await check.has("v", warnings.append) is True
    assert await check.has("v", warnings.append) is True
    assert catalog.capability_calls == ["v", "v"]
    assert len(warnings) == 1


async def test_the_model_name_in_the_warning_is_escaped() -> None:
    check = VisionCheck(FakeCatalog([], fail=True))
    warnings: list[str] = []
    await check.has("m\x1b[2J", warnings.append)
    assert "\x1b" not in warnings[0]
    assert "m\\x1b[2J" in warnings[0]


async def test_a_timeout_is_no_vision() -> None:
    class Hanging(FakeCatalog):
        async def capabilities(self, name: str) -> tuple[str, ...]:
            await asyncio.sleep(5)
            return ("vision",)

    check = VisionCheck(Hanging([]), timeout=0.05)
    warnings: list[str] = []
    assert await check.has("m", warnings.append) is False
    assert len(warnings) == 1


async def test_no_catalog_means_no_vision_without_a_warning() -> None:
    warnings: list[str] = []
    assert await VisionCheck(None).has("m", warnings.append) is False
    assert warnings == []


async def test_an_unexpected_error_is_no_vision_too() -> None:
    class Broken(FakeCatalog):
        async def capabilities(self, name: str) -> tuple[str, ...]:
            raise RuntimeError("bug")

    assert await VisionCheck(Broken([])).has("m") is False


async def test_set_overrides_the_answer() -> None:
    check = VisionCheck(FakeCatalog([]))
    check.set("m", True)
    assert await check.has("m") is True

