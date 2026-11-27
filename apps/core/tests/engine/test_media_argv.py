"""The ffmpeg command lines of assemble_video: one per scene and the final join. Built only from numbers, the motion
enum and fixed names."""

from __future__ import annotations

import re

import pytest
from core.engine import media
from core.engine.media import final_argv, plan_timeline, scene_argv

FF = "/x/ffmpeg"
HEAD = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
TAIL = "d=1:s=1080x1920:fps=30,format=yuv420p"
SCALE = "scale=2484:4416:flags=lanczos"


def _filter(argv: list[str]) -> str:
    return argv[argv.index("-filter_complex") + 1]


def test_a_scene_run_starts_with_the_safe_flags_and_reads_only_the_frame() -> None:
    argv = scene_argv(FF, 0, 3.0, "zoom-in", False)
    assert argv[:6] == HEAD
    assert argv[6:16] == ["-protocol_whitelist", "file", "-loop", "1", "-framerate", "30", "-t", "3.000", "-i", "frame-0.png"]
    assert argv[-1] == "clip-0.mp4"
    assert argv[argv.index("-frames:v") + 1] == "90"
    assert argv[argv.index("-map") + 1] == "[v]"
    assert argv[argv.index("-c:v") + 1 : argv.index("-c:v") + 7] == ["libx264", "-preset", "medium", "-crf", "14", "-pix_fmt"]


@pytest.mark.parametrize(
    "motion, z, x",
    [
        ("zoom-in", "1+0.15*on/90", "iw/2-(iw/zoom/2)"),
        ("zoom-out", "1.15-0.15*on/90", "iw/2-(iw/zoom/2)"),
        ("pan-left", "1.15", "(iw-iw/zoom)*(1-on/90)"),
        ("pan-right", "1.15", "(iw-iw/zoom)*on/90"),
        ("none", "1", "iw/2-(iw/zoom/2)"),
    ],
)
def test_the_zoompan_expression_of_each_motion(motion: str, z: str, x: str) -> None:
    expected = f"[0:v]{SCALE},zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':{TAIL}[v]"
    assert _filter(scene_argv(FF, 0, 3.0, motion, False)) == expected


def test_a_caption_is_overlaid_from_a_second_looped_input() -> None:
    argv = scene_argv(FF, 2, 2.0, "none", True)
    second = argv[16:26]
    assert second == ["-protocol_whitelist", "file", "-loop", "1", "-framerate", "30", "-t", "2.000", "-i", "cap-2.png"]
    assert _filter(argv).endswith(f"{TAIL}[bg];[bg][1:v]overlay=0:0:format=auto,format=yuv420p[v]")
    assert argv[-1] == "clip-2.mp4" and argv[argv.index("-frames:v") + 1] == "60"


def test_every_input_of_every_scene_run_has_the_protocol_whitelist() -> None:
    for caption in (False, True):
        argv = scene_argv(FF, 0, 2.5, "zoom-in", caption)
        assert argv.count("-protocol_whitelist") == argv.count("-i") == (2 if caption else 1)
        for at, arg in enumerate(argv):
            if arg == "-i":
                assert argv[at - 8 : at - 6] == ["-protocol_whitelist", "file"]


def test_the_frame_count_is_the_rounded_length_times_the_rate() -> None:
    argv = scene_argv(FF, 1, 3.3, "zoom-in", False)
    assert argv[argv.index("-frames:v") + 1] == "99"
    assert "1+0.15*on/99" in _filter(argv)


def test_the_hostile_caption_cannot_reach_any_argument() -> None:
    safe = re.compile(r"^[A-Za-z0-9_.:=\-\[\]'*+/()]*$")
    arguments = [*scene_argv(FF, 0, 3.0, "pan-left", True), *final_argv(FF, plan_timeline([3, 3], 0.5), ["nar-0.wav", None], "music.mp3", "out.mp4")]
    for arg in arguments:
        if arg.startswith("-") or arg == FF:
            continue
        for part in re.split(r"[;,]", arg):
            assert safe.match(part.replace("[", "").replace("]", "")), part
    assert not any("%" in a or "\\" in a or "{" in a or '"' in a for a in arguments)


def _inputs(argv: list[str]) -> list[str]:
    return [argv[at + 1] for at, arg in enumerate(argv) if arg == "-i"]


def test_final_join_of_three_scenes_with_a_crossfade_chain_and_music() -> None:
    timeline = plan_timeline([2.0, 3.0, 4.0], 0.5)
    argv = final_argv(FF, timeline, ["nar-0.wav", "nar-1.wav", "nar-2.wav"], "music.mp3", "out.mp4")
    assert argv[:6] == HEAD
    assert _inputs(argv) == ["clip-0.mp4", "clip-1.mp4", "clip-2.mp4", "nar-0.wav", "nar-1.wav", "nar-2.wav", "music.mp3"]
    assert argv.count("-protocol_whitelist") == 7
    assert argv[argv.index("-stream_loop") - 2 : argv.index("-stream_loop") + 4] == [
        "-protocol_whitelist", "file", "-stream_loop", "-1", "-i", "music.mp3",
    ]  # fmt: skip
    assert _filter(argv) == (
        "[0:v][1:v]xfade=transition=fade:duration=0.500:offset=2.000[x1];"
        "[x1][2:v]xfade=transition=fade:duration=0.500:offset=5.500[v];"
        "[3:a]adelay=0:all=1,aresample=48000[a0];"
        "[4:a]adelay=2500:all=1,aresample=48000[a1];"
        "[5:a]adelay=6000:all=1,aresample=48000[a2];"
        "[a0][a1][a2]amix=inputs=3:normalize=0[voice];"
        "[6:a]aresample=48000,volume=-18dB,afade=t=out:st=8.000:d=2.000,atrim=0:10.000[bed];"
        "[voice][bed]amix=inputs=2:normalize=0,atrim=0:10.000[aout]"
    )
    for flag, value in [
        ("-map", "[v]"), ("-c:v", "libx264"), ("-preset", "medium"), ("-crf", "20"), ("-maxrate", "6M"),
        ("-bufsize", "12M"), ("-pix_fmt", "yuv420p"), ("-c:a", "aac"), ("-b:a", "192k"), ("-ar", "48000"),
        ("-ac", "2"), ("-movflags", "+faststart"), ("-t", "10.000"),
    ]:  # fmt: skip
        assert argv[argv.index(flag) + 1] == value
    assert argv[-1] == "out.mp4"


def test_the_two_scene_example_puts_the_second_narration_at_3_3_seconds() -> None:
    argv = final_argv(FF, plan_timeline([2.8, 4.3], 0.5), ["nar-0.wav", "nar-1.wav"], None, "out.mp4")
    graph = _filter(argv)
    assert "[3:a]adelay=3300:all=1,aresample=48000[a1]" in graph
    assert "xfade=transition=fade:duration=0.500:offset=2.800[v]" in graph
    assert argv[argv.index("-t") + 1] == "7.600"


def test_without_music_the_voice_mix_is_the_audio_output() -> None:
    graph = _filter(final_argv(FF, plan_timeline([2.0, 3.0], 0.5), ["a.wav", "b.wav"], None, "out.mp4"))
    assert graph.endswith("[a0][a1]amix=inputs=2:normalize=0,atrim=0:5.500[aout]")
    assert "[voice]" not in graph and "bed" not in graph


def test_scenes_without_audio_are_left_out_of_the_voice_mix() -> None:
    argv = final_argv(FF, plan_timeline([2.0, 3.0], 0.5), [None, "nar-1.wav"], None, "out.mp4")
    assert _inputs(argv) == ["clip-0.mp4", "clip-1.mp4", "nar-1.wav"]
    graph = _filter(argv)
    assert "[2:a]adelay=2500:all=1,aresample=48000[a1]" in graph
    assert "[a1]amix=inputs=1:normalize=0" in graph


def test_music_alone_is_trimmed_and_faded_to_the_total() -> None:
    argv = final_argv(FF, plan_timeline([5.0], 0.5), [None], "music.m4a", "out.mp4")
    assert _filter(argv) == (
        "[0:v]null[v];[1:a]aresample=48000,volume=-18dB,afade=t=out:st=3.000:d=2.000,atrim=0:5.000[aout]"
    )
    assert "-stream_loop" in argv


def test_no_audio_at_all_has_no_audio_stream() -> None:
    argv = final_argv(FF, plan_timeline([5.0], 0.5), [None], None, "out.mp4")
    assert _filter(argv) == "[0:v]null[v]"
    assert "-an" in argv and "-c:a" not in argv and "[aout]" not in argv


def test_a_zero_crossfade_joins_with_hard_cuts() -> None:
    argv = final_argv(FF, plan_timeline([2.0, 3.0, 4.0], 0.0), [None, None, None], None, "out.mp4")
    assert _filter(argv) == "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]"
    assert argv[argv.index("-t") + 1] == "9.000"


def test_a_music_fade_never_starts_before_zero() -> None:
    graph = _filter(final_argv(FF, plan_timeline([1.0], 0.5), [None], "music.mp3", "out.mp4"))
    assert "afade=t=out:st=0.000:d=1.000" in graph


def test_the_constants_the_filters_use() -> None:
    assert (media.MUSIC_DB, media.MUSIC_FADE_S) == (-18, 2.0)
