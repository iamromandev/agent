from __future__ import annotations

import pytest
from src.engine.provider.models import CatalogError, ModelInfo, launch_model, pick_launch_model, resolve_window

from tests.fakes import FakeCatalog


@pytest.mark.parametrize(
    ("length", "cap", "expected"),
    [(40960, 8192, 8192), (4096, 8192, 4096), (None, 8192, 8192), (0, 8192, 8192), (8192, 8192, 8192)],
)
def test_the_window_is_the_smaller_of_the_models_length_and_the_cap(length: int | None, cap: int, expected: int) -> None:
    assert resolve_window(length, cap) == expected


def test_a_flag_beats_the_environment_the_project_key_and_the_default() -> None:
    assert pick_launch_model("x", "y", "z", "d", ["x", "y", "z", "d"]) == ("x", [])


def test_the_environment_beats_the_project_key() -> None:
    assert pick_launch_model(None, "y", "z", "d", ["z"]) == ("y", [])


def test_the_project_key_beats_the_default_when_installed() -> None:
    assert pick_launch_model(None, None, "z", "d", ["z", "d"]) == ("z", [])


def test_a_project_key_matches_the_latest_tag_like_check_does() -> None:
    assert pick_launch_model(None, None, "z", "d", ["z:latest"]) == ("z", [])


def test_a_project_key_that_is_not_installed_falls_back_with_an_escaped_warning() -> None:
    model, warnings = pick_launch_model(None, None, "z\x1b[2J", "d", ["a"])
    assert model == "d"
    assert len(warnings) == 1
    assert "\x1b" not in warnings[0]
    assert "z\\x1b[2J" in warnings[0]


@pytest.mark.parametrize("bad", [5, ["a"], {"a": 1}, True, ""])
def test_a_project_key_that_is_not_a_string_falls_back_with_a_warning(bad: object) -> None:
    model, warnings = pick_launch_model(None, None, bad, "d", ["a"])
    assert model == "d"
    assert len(warnings) == 1


def test_no_project_key_is_no_warning() -> None:
    assert pick_launch_model(None, None, None, "d", ["a"]) == ("d", [])


def test_an_unreachable_ollama_takes_the_project_key_as_it_is() -> None:
    assert pick_launch_model(None, None, "z", "d", None) == ("z", [])


async def test_the_fake_catalog_lists_and_reports_lengths_and_can_fail() -> None:
    catalog = FakeCatalog([ModelInfo("a", size=5), ModelInfo("b")], lengths={"a": 4096})
    assert [m.name for m in await catalog.list()] == ["a", "b"]
    assert await catalog.context_length("a") == 4096
    assert await catalog.context_length("b") is None
    catalog.fail = True
    with pytest.raises(CatalogError):
        await catalog.list()


# launch_model


class Cfg:
    """The bits of Settings that launch_model reads."""

    def __init__(self, model: str = "default", num_ctx: int = 8192, explicit: bool = False) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.model_fields_set = {"model"} if explicit else set()


async def test_launch_uses_the_flag_first_and_caps_its_window() -> None:
    catalog = FakeCatalog(["x", "y", "z"], lengths={"x": 40960})

    model, window, warnings = await launch_model(Cfg(model="y", explicit=True), "x", {"model": "z"}, catalog)

    assert (model, window, warnings) == ("x", 8192, [])


async def test_launch_uses_an_explicit_environment_model_before_the_project_key() -> None:
    catalog = FakeCatalog(["y", "z"], lengths={"y": 4096})

    model, window, warnings = await launch_model(Cfg(model="y", explicit=True), None, {"model": "z"}, catalog)

    assert (model, window, warnings) == ("y", 4096, [])


async def test_launch_uses_the_project_key_when_nothing_is_explicit() -> None:
    catalog = FakeCatalog(["default", "z"], lengths={"z": 2048})

    assert await launch_model(Cfg(), None, {"model": "z"}, catalog) == ("z", 2048, [])


async def test_launch_falls_back_to_the_default_with_a_warning_when_the_project_model_is_not_installed() -> None:
    catalog = FakeCatalog(["default"], lengths={"default": 4096})

    model, window, warnings = await launch_model(Cfg(), None, {"model": "gone"}, catalog)

    assert (model, window) == ("default", 4096)
    assert len(warnings) == 1
    assert "gone" in warnings[0]


async def test_launch_without_a_project_file_uses_the_default_and_asks_nothing_about_installed_models() -> None:
    catalog = FakeCatalog(["default"], lengths={"default": 100000})

    assert await launch_model(Cfg(), None, None, catalog) == ("default", 8192, [])


async def test_a_failed_length_lookup_gives_the_cap_and_a_warning() -> None:
    catalog = FakeCatalog(["m"])
    catalog.fail_length.add("m")

    model, window, warnings = await launch_model(Cfg(model="m", explicit=True), None, None, catalog)

    assert (model, window) == ("m", 8192)
    assert warnings == ["Could not read the context length of m; using AGENT_NUM_CTX (8192)"]


async def test_an_unknown_length_is_the_cap_without_a_warning() -> None:
    assert await launch_model(Cfg(model="m", explicit=True), None, None, FakeCatalog(["m"])) == ("m", 8192, [])


async def test_an_unreachable_catalog_takes_the_project_key_and_warns_once_about_the_length() -> None:
    catalog = FakeCatalog(fail=True)

    model, window, warnings = await launch_model(Cfg(), None, {"model": "z"}, catalog)

    assert (model, window) == ("z", 8192)
    assert len(warnings) == 1
    assert "context length" in warnings[0]
    assert catalog.length_calls == []  # no second wait for a server that did not answer


async def test_a_slow_catalog_is_given_up_on_after_the_timeout() -> None:
    catalog = FakeCatalog(["z"], delay=5)

    model, window, warnings = await launch_model(Cfg(), None, {"model": "z"}, catalog, timeout=0.05)

    assert (model, window) == ("z", 8192)
    assert len(warnings) == 1


async def test_the_warning_escapes_the_model_name() -> None:
    catalog = FakeCatalog(["a\x1b[2J"])
    catalog.fail_length.add("a\x1b[2J")

    _, _, warnings = await launch_model(Cfg(model="a\x1b[2J", explicit=True), None, None, catalog)

    assert "\x1b" not in warnings[0]


async def test_a_project_data_that_is_not_an_object_is_no_key() -> None:
    assert await launch_model(Cfg(model="m", explicit=True), None, None, FakeCatalog(["m"])) == ("m", 8192, [])


async def launch_notices(cfg: Cfg, flag: str | None, project: object, catalog: FakeCatalog) -> tuple[str, int, list[str], list[str]]:
    notices: list[str] = []
    model, window, warnings = await launch_model(cfg, flag, {"model": project}, catalog, notices=notices)
    return model, window, warnings, notices


async def test_a_project_key_that_decided_the_model_is_announced_with_the_file_it_came_from() -> None:
    model, _, _, notices = await launch_notices(Cfg(), None, "z", FakeCatalog(["default", "z"]))

    assert model == "z"
    assert notices == ["Model z (from .agent/settings.json)"]


async def test_the_project_notice_escapes_the_name() -> None:
    _, _, _, notices = await launch_notices(Cfg(), None, "z\x1b[2J", FakeCatalog(["z\x1b[2J"]))

    assert len(notices) == 1
    assert "\x1b" not in notices[0]
    assert "z\\x1b[2J" in notices[0]


async def test_no_notice_when_the_flag_the_environment_or_the_default_decided() -> None:
    catalog = FakeCatalog(["default", "z", "y", "x"])

    assert (await launch_notices(Cfg(), "x", "z", catalog))[3] == []
    assert (await launch_notices(Cfg(model="y", explicit=True), None, "z", catalog))[3] == []
    assert (await launch_notices(Cfg(), None, "gone", catalog))[3] == []  # not installed: the default, and a warning
    assert (await launch_notices(Cfg(), None, 5, catalog))[3] == []


async def test_the_project_notice_is_also_given_when_ollama_cannot_be_reached() -> None:
    _, _, _, notices = await launch_notices(Cfg(), None, "z", FakeCatalog(fail=True))

    assert notices == ["Model z (from .agent/settings.json)"]


async def test_a_project_key_is_stripped_before_it_is_matched_and_used() -> None:
    model, _, warnings, notices = await launch_notices(Cfg(), None, "  z \n", FakeCatalog(["z"]))

    assert (model, warnings) == ("z", [])
    assert notices == ["Model z (from .agent/settings.json)"]


@pytest.mark.parametrize("blank", ["", "   "])
async def test_a_blank_explicit_environment_model_is_unset(blank: str) -> None:
    catalog = FakeCatalog(["default", "z"])

    model, _, _, notices = await launch_notices(Cfg(model=blank, explicit=True), None, "z", catalog)

    assert model == "z"
    assert notices != []


async def test_a_huge_project_key_with_an_unreachable_ollama_is_cut_in_every_line() -> None:
    huge = "m" * 5000

    _, _, warnings, notices = await launch_notices(Cfg(), None, huge, FakeCatalog(fail=True))

    for line in (*warnings, *notices):
        assert len(line) < 400
        assert "…" in line


async def test_a_huge_flag_with_an_unreachable_ollama_is_cut_in_the_warning() -> None:
    _, _, warnings = await launch_model(Cfg(), "m" * 5000, None, FakeCatalog(fail=True))

    assert len(warnings) == 1
    assert len(warnings[0]) < 400
    assert "…" in warnings[0]


def test_a_huge_project_key_that_is_not_installed_is_cut_in_the_warning() -> None:
    _, warnings = pick_launch_model(None, None, "m" * 5000, "d", ["a"])

    assert len(warnings[0]) < 400
    assert "…" in warnings[0]


def test_a_huge_non_string_project_key_is_cut_in_the_warning() -> None:
    _, warnings = pick_launch_model(None, None, ["m" * 5000], "d", ["a"])

    assert len(warnings[0]) < 400


async def test_a_slow_lookup_calls_the_slow_note_once_with_the_model_asked_about() -> None:
    seen: list[str] = []

    model, _, _ = await launch_model(
        Cfg(model="m", explicit=True), None, None, FakeCatalog(["m"], delay=0.2), on_slow=seen.append, slow_after=0.02
    )

    assert model == "m"
    assert seen == ["m"]


async def test_a_quick_lookup_makes_no_slow_note() -> None:
    seen: list[str] = []

    await launch_model(Cfg(model="m", explicit=True), None, None, FakeCatalog(["m"]), on_slow=seen.append, slow_after=0.5)

    assert seen == []


async def test_the_slow_note_names_the_project_key_when_that_is_what_is_asked_about() -> None:
    seen: list[str] = []

    await launch_model(Cfg(), None, {"model": " z "}, FakeCatalog(["z"], delay=0.2), on_slow=seen.append, slow_after=0.02)

    assert seen == ["z"]
