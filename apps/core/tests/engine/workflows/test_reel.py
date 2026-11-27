"""The built-in `reel` workflow: it loads through the real loader and plans the calls the spec names (AC11)."""

from __future__ import annotations

import json
from pathlib import Path

from core.engine.workflows.loader import builtin_dir, load_workflows
from core.engine.workflows.schema import validate

from tests.engine.workflows.test_runner import RUN, answer, build, run

MOTIONS = ["zoom-in", "zoom-out", "pan-left", "pan-right", "none"]


def story(n: int) -> str:
    return json.dumps(
        {
            "title": "Cats",
            "scenes": [
                {
                    "narration": f"Narration number {i} of the story.",
                    "caption": f"Caption {i}",
                    "image_prompt": f"A cat, scene {i}, portrait",
                    "motion": MOTIONS[i % 5],
                }
                for i in range(1, n + 1)
            ],
        }
    )


def reel(tmp_path: Path):
    loaded = load_workflows(tmp_path, {"XDG_CONFIG_HOME": str(tmp_path / "none")})
    assert loaded.warnings == []
    return loaded.items["reel"]


def test_reel_ships_with_the_agent_and_loads_without_warnings(tmp_path: Path) -> None:
    assert (builtin_dir() / "reel.json").is_file()
    wf = reel(tmp_path)

    assert wf.source == "builtin"
    assert wf.folder == "reels/{run}"
    assert [s.id for s in wf.steps] == ["story", "images", "voice", "video"]
    assert wf.input_help


def test_the_story_schema_wants_three_to_eight_scenes_with_the_spec_fields(tmp_path: Path) -> None:
    schema = reel(tmp_path).steps[0].schema
    assert schema is not None

    assert validate(schema, json.loads(story(3))) is None
    assert validate(schema, json.loads(story(8))) is None
    assert validate(schema, json.loads(story(2))) is not None
    assert validate(schema, json.loads(story(9))) is not None
    scene = json.loads(story(3))
    scene["scenes"][0]["motion"] = "spin"
    assert validate(schema, scene) is not None
    scene = json.loads(story(3))
    scene["title"] = "x" * 61
    assert validate(schema, scene) is not None
    scene = json.loads(story(3))
    scene["scenes"][0]["narration"] = ""
    assert validate(schema, scene) is not None
    scene = json.loads(story(3))
    scene["scenes"][0]["caption"] = "c" * 61
    assert validate(schema, scene) is not None
    scene = json.loads(story(3))
    scene["scenes"][0]["image_prompt"] = "p" * 501
    assert validate(schema, scene) is not None


def _scene_schema(tmp_path: Path) -> dict:
    schema = reel(tmp_path).steps[0].schema
    assert schema is not None
    return schema["properties"]["scenes"]["items"]["properties"]


def test_a_narration_is_long_enough_to_make_a_second_of_audio(tmp_path: Path) -> None:
    narration = _scene_schema(tmp_path)["narration"]
    assert narration["minLength"] >= 30
    scene = json.loads(story(3))
    scene["scenes"][0]["narration"] = "Hi."
    assert validate(reel(tmp_path).steps[0].schema, scene) is not None


def test_every_caption_the_schema_allows_fits_four_lines_of_the_video(tmp_path: Path) -> None:
    from core.engine.media import format_text

    limit = _scene_schema(tmp_path)["caption"]["maxLength"]
    assert limit <= 64  # five lines need at least 65 characters: two line pairs of 32 plus one
    cases = [" ".join(["a" * n] * 6)[:limit] for n in range(1, 40)]
    cases += [" ".join(("a" * 17, "b" * 16) * 4)[:limit], "x" * limit, (("a" * 31 + " ") * 3 + "b" * 31)[:limit]]
    import random

    rng = random.Random(7)
    for _ in range(20000):
        words, size = [], 0
        while size < limit:
            word = "w" * rng.randint(1, 40)
            words.append(word)
            size += len(word) + 1
        cases.append(" ".join(words)[:limit])
    for text in cases:
        assert len(text) <= limit
        assert format_text(text) is not None, text


def test_the_story_prompt_states_the_narration_and_caption_limits(tmp_path: Path) -> None:
    step = reel(tmp_path).steps[0]
    props = _scene_schema(tmp_path)
    assert f"at most {props['caption']['maxLength']} characters" in step.prompt
    assert f"at least {props['narration']['minLength']} characters" in step.prompt


async def test_the_planned_calls_pass_the_real_tools_argument_checks(tmp_path: Path) -> None:
    from core.engine.tools.base import check_args
    from core.engine.tools.image import GenerateImageTool
    from core.engine.tools.media import AssembleVideoTool, SpeakTool

    real = {t.name: t for t in (GenerateImageTool, SpeakTool, AssembleVideoTool)}
    env = build(tmp_path, [answer(story(4))])
    env.approved.clear()

    result = await run(env, reel(tmp_path), "cats")

    assert result.ok, result.line
    calls = env.approved[0].calls
    assert {c.tool for c in calls} == set(real)
    for call in calls:
        assert check_args(real[call.tool].parameters, call.args) is None, call.key


async def test_a_three_scene_story_plans_seven_calls_with_the_stated_arguments(tmp_path: Path) -> None:
    env = build(tmp_path, [answer(story(3))])
    env.approved.clear()

    result = await run(env, reel(tmp_path), "cats")

    assert result.ok, result.line
    folder = f"reels/{RUN}"
    assert result.folder == folder
    assert env.img.calls == [
        {
            "prompt": f"A cat, scene {i}, portrait",
            "path": f"{folder}/scene-{i}.png",
            "width": 576,
            "height": 1024,
        }
        for i in (1, 2, 3)
    ]
    assert env.voice.calls == [
        {"text": f"Narration number {i} of the story.", "path": f"{folder}/scene-{i}.wav"} for i in (1, 2, 3)
    ]
    assert env.video.calls == [
        {
            "path": f"{folder}/reel.mp4",
            "crossfade": 0.5,
            "scenes": [
                {
                    "image": f"{folder}/scene-{i}.png",
                    "audio": f"{folder}/scene-{i}.wav",
                    "motion": MOTIONS[i % 5],
                    "text": f"Caption {i}",
                }
                for i in (1, 2, 3)
            ],
        }
    ]
    assert len(env.approved[0].calls) == 7
