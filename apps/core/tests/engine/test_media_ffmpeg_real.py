"""Tiny reels rendered by the real ffmpeg and the real media helper, to prove that the plan, the commands and the
files agree: the video track and the file are as long as `plan_timeline` says, within one frame. Skipped without the
`media` extra (ffmpeg from imageio-ffmpeg, Pillow) or off Apple Silicon. Scenes are a fraction of a second long."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import pytest
from core.engine.media import FPS, FRAME, final_argv, find_ffmpeg, mp4_info, plan_timeline, scene_argv
from core.testing.image_fakes import png

# Loaded through importorskip, not `import`: CI has no media extra, and ty must not need Pillow to check this file.
pytest.importorskip("PIL.Image")
FFMPEG = find_ffmpeg()
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="no ffmpeg (media extra on Apple Silicon)")

HELPER = Path(__file__).resolve().parents[2] / "core" / "engine" / "media_helper.py"
ONE_FRAME = 1 / FPS + 1e-3


def _helper(folder: Path, *args: object) -> None:
    argv = [sys.executable, "-I", str(HELPER), *map(str, args)]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=folder)
    assert done.returncode == 0, done.stderr


def _run(folder: Path, argv: list[str]) -> None:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=folder)
    assert done.returncode == 0, done.stderr[-2000:]


def _wav(path: Path, seconds: float, rate: int = 24000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x10\x10" * channels * int(rate * seconds))


def _render(
    folder: Path, lengths: list[float], crossfade: float, spoken: list[bool], music: float | None = None,
    captions: tuple[int, ...] = (),
) -> tuple[float, float, float]:
    """Render the reel and return (plan total, video track seconds, file seconds)."""
    assert FFMPEG is not None
    plan = plan_timeline(lengths, crossfade)
    narrations: list[str | None] = []
    for i, length in enumerate(lengths):
        (folder / f"in-{i}.png").write_bytes(png(60 + i, 40))
        _helper(folder, "frame", "--in", f"in-{i}.png", "--out", f"frame-{i}.png", "--width", FRAME[0], "--height", FRAME[1])
        if i in captions:
            (folder / f"cap-{i}.txt").write_text("A short caption\nover two lines")
            _helper(folder, "caption", "--text-file", f"cap-{i}.txt", "--out", f"cap-{i}.png")
        if spoken[i]:
            _wav(folder / f"nar-{i}.wav", length)
            narrations.append(f"nar-{i}.wav")
        else:
            narrations.append(None)
        _run(folder, scene_argv(FFMPEG, i, plan.clip_lengths[i], "zoom-in", i in captions))
    music_name = None
    if music is not None:
        music_name = "music.wav"
        _wav(folder / music_name, music, 44100, 2)
    _run(folder, final_argv(FFMPEG, plan, narrations, music_name, "out.mp4"))
    info = mp4_info((folder / "out.mp4").read_bytes())
    assert info is not None and info.faststart and (info.width, info.height) == FRAME
    return plan.total, info.video_seconds, info.seconds


def _agrees(result: tuple[float, float, float]) -> None:
    total, video, whole = result
    assert abs(video - total) <= ONE_FRAME, f"video track {video:.3f} s, plan {total:.3f} s"
    assert abs(whole - total) <= ONE_FRAME, f"file {whole:.3f} s, plan {total:.3f} s"


def test_three_narrated_scenes_with_a_crossfade_of_half_a_second(tmp_path: Path) -> None:
    _agrees(_render(tmp_path, [0.6, 0.5, 0.7], 0.5, [True, True, True], captions=(1,)))


def test_the_concat_path_with_lengths_off_the_frame_grid_does_not_drift(tmp_path: Path) -> None:
    # 5 x 1.0166 s is 5.083 s on paper but 5 x 30 frames = 5.0 s when each clip is rounded to frames
    _agrees(_render(tmp_path, [1.0166] * 5, 0.0, [True] * 5))


def test_a_silent_scene_and_music_shorter_than_the_video(tmp_path: Path) -> None:
    _agrees(_render(tmp_path, [0.6, 0.5, 0.6], 0.5, [True, False, True], music=0.25))
