from __future__ import annotations

import pytest
from core.engine.provider.models import launch_model, pick_launch_model
from core.testing.fakes import FakeCatalog


def test_a_flag_beats_the_environment_the_project_key_and_the_default() -> None:
    assert pick_launch_model("x", "y", "z", "d", ["x", "y", "z", "d"]) == ("x", [])


def test_a_project_key_matches_the_latest_tag_like_check_does() -> None:
    assert pick_launch_model(None, None, "z", "d", ["z:latest"]) == ("z", [])


def test_a_project_key_that_is_not_installed_falls_back_with_an_escaped_warning() -> None:
    model, warnings = pick_launch_model(None, None, "z\x1b[2J", "d", ["a"])
    assert model == "d"
    assert len(warnings) == 1
    assert "\x1b" not in warnings[0]
    assert "z\\x1b[2J" in warnings[0]


@pytest.mark.parametrize("bad", [{"a": 1}])
def test_a_project_key_that_is_not_a_string_falls_back_with_a_warning(bad: object) -> None:
    model, warnings = pick_launch_model(None, None, bad, "d", ["a"])
    assert model == "d"
    assert len(warnings) == 1


# launch_model


class Cfg:
    """The bits of Settings that launch_model reads."""

    def __init__(self, model: str = "default", num_ctx: int = 8192, explicit: bool = False) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.model_fields_set = {"model"} if explicit else set()


async def test_a_failed_length_lookup_gives_the_cap_and_a_warning() -> None:
    catalog = FakeCatalog(["m"])
    catalog.fail_length.add("m")

    model, window, warnings = await launch_model(Cfg(model="m", explicit=True), None, None, catalog)

    assert (model, window) == ("m", 8192)
    assert warnings == ["Could not read the context length of m; using AGENT_NUM_CTX (8192)"]


async def launch_notices(cfg: Cfg, flag: str | None, project: object, catalog: FakeCatalog) -> tuple[str, int, list[str], list[str]]:
    notices: list[str] = []
    model, window, warnings = await launch_model(cfg, flag, {"model": project}, catalog, notices=notices)
    return model, window, warnings, notices


async def test_the_project_notice_is_also_given_when_ollama_cannot_be_reached() -> None:
    _, _, _, notices = await launch_notices(Cfg(), None, "z", FakeCatalog(fail=True))

    assert notices == ["Model z (from .agent/settings.json)"]


async def test_the_slow_note_names_the_project_key_when_that_is_what_is_asked_about() -> None:
    seen: list[str] = []

    await launch_model(Cfg(), None, {"model": " z "}, FakeCatalog(["z"], delay=0.2), on_slow=seen.append, slow_after=0.02)

    assert seen == ["z"]
