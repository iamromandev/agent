"""Media helpers: sniffing audio, reading WAV headers, finding the speech engine."""

from __future__ import annotations

import struct
import sys
import types
from pathlib import Path

import pytest
from core.engine import media
from core.engine.media import DEFAULT_VOICE, VOICES, WavInfo, find_speech, sniff_audio, wav_info
from core.testing.media_fakes import mp4_bytes


def wav(
    seconds: float = 1.0, rate: int = 24000, channels: int = 1, bits: int = 16, fmt: int = 1, extensible: bool = False,
    extra_chunk: bool = False,
) -> bytes:
    width = bits // 8
    frames = int(seconds * rate)
    data = b"\x00" * (frames * channels * width)
    if extensible:
        sub = struct.pack("<H", fmt) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        body = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * channels * width, channels * width, bits)
        body += struct.pack("<HHI", 22, bits, 0) + sub
    else:
        body = struct.pack("<HHIIHH", fmt, channels, rate, rate * channels * width, channels * width, bits)
    chunks = b"fmt " + struct.pack("<I", len(body)) + body
    if extra_chunk:
        chunks += b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\x00"  # odd size: padded
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_voices_and_default() -> None:
    assert DEFAULT_VOICE == "af_heart" and DEFAULT_VOICE in VOICES
    assert len(VOICES) == 28 and "bm_lewis" in VOICES
    assert media.MAX_TEXT == 4000 and media.MAX_AUDIO_BYTES == 50 * 1024 * 1024


@pytest.mark.parametrize(
    "head, kind",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"ID3\x04\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x00", "mp3"),
        (b"\xff\xe0", None),  # layer 00 is reserved
        (b"\x00\x00\x00\x20ftypM4A \x00", "m4a"),
        (b"\x00\x00\x00\x20ftypmp42", "m4a"),
        (b"\x00\x00\x00\x20ftypisom", "m4a"),
        (b"\xff\x1f", None),
        (b"\xff\xea\x90\x00", None),  # reserved MPEG version 01
        (b"\xff\xf9\x90\x00", None),  # reserved layer 00
        (b"\xff\xe2\x00\x00", "mp3"),  # MPEG 2.5 layer 3
        (b"RIFF\x00\x00\x00\x00AVI ", None),
        (b"\x00\x00\x00\x20ftypheic", None),
        (b"", None),
        (b"text", None),
    ],
)
def test_sniff_audio(head: bytes, kind: str | None) -> None:
    assert sniff_audio(head) == kind


def test_wav_info_reads_pcm16() -> None:
    assert wav_info(wav(2.0)) == WavInfo(24000, 1, 16, 2.0)


@pytest.mark.parametrize("bits, fmt", [(16, 1), (24, 1), (32, 1), (32, 3)])
def test_wav_info_accepts_pcm_and_float_formats(bits: int, fmt: int) -> None:
    info = wav_info(wav(0.5, rate=44100, channels=2, bits=bits, fmt=fmt))
    assert info == WavInfo(44100, 2, bits, 0.5)


@pytest.mark.parametrize("bits, fmt", [(16, 1), (32, 3)])
def test_wav_info_accepts_extensible(bits: int, fmt: int) -> None:
    assert wav_info(wav(1.0, bits=bits, fmt=fmt, extensible=True)) == WavInfo(24000, 1, bits, 1.0)


def test_wav_info_walks_past_other_chunks() -> None:
    assert wav_info(wav(1.0, extra_chunk=True)) == WavInfo(24000, 1, 16, 1.0)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"RIFF",
        b"not a wav at all, just text" * 3,
        wav(1.0)[:30],  # cut inside fmt
        wav(1.0, fmt=2),  # ADPCM
        wav(1.0, bits=8),
        wav(0.0),  # no audio
        wav(1.0, rate=0),
        wav(1.0, channels=0),
    ],
)
def test_wav_info_is_none_when_malformed_or_unsupported(data: bytes) -> None:
    assert wav_info(data) is None


def test_wav_info_uses_the_data_actually_present_when_the_header_overclaims() -> None:
    data = bytearray(wav(1.0))
    cut = bytes(data[: 44 + 24000])  # half the samples
    info = wav_info(cut)
    assert info is not None and info.seconds == pytest.approx(0.5, abs=0.01)


def test_wav_info_without_a_data_chunk_is_none() -> None:
    full = wav(1.0)
    assert wav_info(full[: full.index(b"data")]) is None


def test_wav_info_is_bounded_on_many_tiny_chunks() -> None:
    chunks = b"".join(b"JUNK" + struct.pack("<I", 0) for _ in range(200_000))
    assert wav_info(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks) is None


def test_find_speech_needs_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.importlib.util, "find_spec", lambda name: object())
    assert find_speech("linux", "arm64") is None
    assert find_speech("darwin", "x86_64") is None
    assert find_speech("darwin", "arm64") == media.SPEECH_HELPER


@pytest.mark.parametrize("error", [ValueError("spec is None"), ImportError("broken parent")])
def test_find_speech_survives_find_spec_errors(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def boom(name: str) -> None:
        raise error

    monkeypatch.setattr(media.importlib.util, "find_spec", boom)
    assert find_speech("darwin", "arm64") is None


@pytest.mark.parametrize("missing", ["mlx_audio", "misaki"])
def test_find_speech_needs_both_packages(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setattr(media.importlib.util, "find_spec", lambda name: None if name == missing else object())
    assert find_speech("darwin", "arm64") is None


def test_find_speech_needs_the_helper_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(media, "SPEECH_HELPER", tmp_path / "nope.py")
    assert find_speech("darwin", "arm64") is None


def test_the_limits_are_the_documented_ones() -> None:
    assert media.FRAME == (1080, 1920) and media.FPS == 30
    assert (media.MAX_SCENES, media.MAX_SCENE_S, media.MAX_TOTAL_S) == (30, 60, 180)
    assert (media.MAX_CAPTION, media.MAX_CAPTION_LINES) == (200, 4)
    assert media.MOTIONS == ("zoom-in", "zoom-out", "pan-left", "pan-right", "none")
    assert media.MAX_IMAGE_BYTES == 20 * 1024 * 1024 and media.MAX_VIDEO_BYTES == 200 * 1024 * 1024


def test_timeline_of_two_scenes_with_a_crossfade() -> None:
    t = media.plan_timeline([2.8, 4.3], 0.5)
    assert t.total == pytest.approx(7.6)
    assert t.starts == pytest.approx((0.0, 3.3))
    assert t.offsets == pytest.approx((2.8,))
    assert t.clip_lengths == pytest.approx((3.3, 4.8))


def test_timeline_clip_lengths_of_three_scenes() -> None:
    t = media.plan_timeline([2.0, 3.0, 4.0], 0.5)
    assert t.clip_lengths == pytest.approx((2.5, 4.0, 4.5))
    assert t.starts == pytest.approx((0.0, 2.5, 6.0))
    assert t.offsets == pytest.approx((2.0, 5.5))
    assert t.total == pytest.approx(10.0)


def test_timeline_of_a_single_scene_is_its_own_length() -> None:
    t = media.plan_timeline([5.0], 0.5)
    assert t.clip_lengths == (5.0,) and t.offsets == () and t.starts == (0.0,) and t.total == 5.0


def test_timeline_without_a_crossfade_is_hard_cuts() -> None:
    t = media.plan_timeline([2.0, 3.0], 0.0)
    assert t.clip_lengths == (2.0, 3.0) and t.starts == (0.0, 2.0) and t.total == 5.0


def _on_grid(value: float) -> bool:
    frames = value * media.FPS
    return abs(frames - round(frames)) < 1e-4


def _frames(t: media.Timeline) -> int:
    return sum(round(c * media.FPS) for c in t.clip_lengths) - round(t.crossfade * media.FPS) * (len(t.clip_lengths) - 1)


def test_timeline_concat_path_total_matches_the_frames_rendered() -> None:
    t = media.plan_timeline([1.0166] * 12, 0.0)
    assert round(t.total * media.FPS) == sum(round(c * media.FPS) for c in t.clip_lengths) == 372
    assert t.starts[11] == pytest.approx(sum(t.clip_lengths[:11]))
    assert all(_on_grid(v) for v in (*t.starts, *t.clip_lengths, *t.offsets, t.total))


def test_timeline_of_a_silent_reel_with_an_off_grid_duration_sits_on_frames() -> None:
    t = media.plan_timeline([1.016] * 30, 0.5)
    assert all(_on_grid(v) for v in (*t.starts, *t.clip_lengths, *t.offsets, t.total, t.crossfade))
    assert round(t.total * media.FPS) == _frames(t)
    assert t.starts[29] == pytest.approx(t.offsets[28] + t.crossfade)


def test_timeline_crossfade_is_a_whole_number_of_frames() -> None:
    t = media.plan_timeline([2.0, 2.0], 0.51)
    assert t.crossfade == pytest.approx(15 / media.FPS)
    assert all(_on_grid(v) for v in (*t.starts, *t.clip_lengths, *t.offsets, t.total))


def test_timeline_never_cuts_a_narration_that_is_a_few_ms_over_a_frame() -> None:
    t = media.plan_timeline([2.0 + 0.004], 0.0)
    assert t.total >= 2.004 and _on_grid(t.total) and t.total == pytest.approx(61 / media.FPS)


def test_format_text_wraps_at_32_characters() -> None:
    text = media.format_text("the quick brown fox jumps over the lazy dog again and again")
    assert text is not None
    lines = text.split("\n")
    assert 1 < len(lines) <= 4 and all(len(line) <= 32 for line in lines)
    assert " ".join(lines) == "the quick brown fox jumps over the lazy dog again and again"


def test_format_text_collapses_whitespace_and_newlines() -> None:
    assert media.format_text("a\n\n  b\tc") == "a b c"
    assert media.format_text("   ") == ""


def test_format_text_breaks_a_word_longer_than_a_line() -> None:
    text = media.format_text("x" * 70)
    assert text is not None and [len(line) for line in text.split("\n")] == [32, 32, 6]


def test_format_text_with_more_than_four_lines_is_none() -> None:
    assert media.format_text("word " * 40) is None
    assert media.format_text("y" * 129) is None
    assert media.format_text("y" * 128) is not None


@pytest.mark.parametrize("version", [0, 1])
def test_mp4_info_reads_the_size_and_length(version: int) -> None:
    info = media.mp4_info(mp4_bytes(1080, 1920, 7.6, version=version))
    assert info is not None and (info.width, info.height, info.faststart) == (1080, 1920, True)
    assert info.seconds == pytest.approx(7.6)


def test_mp4_info_skips_the_audio_track() -> None:
    info = media.mp4_info(mp4_bytes(720, 1280, audio=True))
    assert info is not None and (info.width, info.height) == (720, 1280)


def test_mp4_info_without_faststart_when_mdat_comes_first() -> None:
    info = media.mp4_info(mp4_bytes(faststart=False))
    assert info is not None and info.faststart is False


def test_mp4_info_uses_the_timescale() -> None:
    info = media.mp4_info(mp4_bytes(seconds=2.5, timescale=90000))
    assert info is not None and info.seconds == pytest.approx(2.5)


@pytest.mark.parametrize("cut", [0, 4, 7, 20, 60, 120, 200])
def test_mp4_info_of_a_truncated_file_is_none(cut: int) -> None:
    assert media.mp4_info(mp4_bytes()[:cut]) is None


def test_mp4_info_of_garbage_is_none() -> None:
    assert media.mp4_info(b"") is None
    assert media.mp4_info(b"not an mp4 at all, just text" * 10) is None
    assert media.mp4_info(bytes(range(256)) * 4) is None


def test_mp4_info_of_a_box_that_claims_to_be_huge_is_none() -> None:
    assert media.mp4_info(struct.pack(">I", 0xFFFFFFF0) + b"moov" + b"\0" * 40) is None


def test_mp4_info_is_bounded_on_many_tiny_boxes() -> None:
    assert media.mp4_info((struct.pack(">I", 8) + b"free") * 300_000) is None


@pytest.mark.parametrize("version", [0, 1])
def test_mp4_info_reads_the_video_track_length_from_mdhd_in_its_own_timescale(version: int) -> None:
    info = media.mp4_info(mp4_bytes(seconds=7.6, video_seconds=6.5, version=version))
    assert info is not None and info.seconds == pytest.approx(7.6) and info.video_seconds == pytest.approx(6.5, abs=1e-3)


def test_mp4_info_video_length_skips_the_audio_track() -> None:
    info = media.mp4_info(mp4_bytes(seconds=9.0, video_seconds=4.0, audio=True))
    assert info is not None and info.video_seconds == pytest.approx(4.0, abs=1e-3)


@pytest.mark.parametrize("version", [0, 1])
def test_mp4_info_video_length_falls_back_to_tkhd_in_the_movie_timescale(version: int) -> None:
    info = media.mp4_info(mp4_bytes(seconds=7.6, video_seconds=6.5, version=version, mdhd=False, timescale=600))
    assert info is not None and info.video_seconds == pytest.approx(6.5, abs=1e-2)


def test_mp4_info_is_none_without_a_video_track_length() -> None:
    data = mp4_bytes(mdhd=False).replace(b"vide", b"soun")  # no track says it is video
    assert media.mp4_info(data) is None


def test_mp4_info_of_a_zero_track_timescale_is_none() -> None:
    assert media.mp4_info(mp4_bytes(track_timescale=0)) is None


def test_mp4_info_reads_a_header_cut_after_moov_when_given_the_file_size() -> None:
    data = mp4_bytes()
    head = data[: data.index(b"mdat") - 4 + 8 + 10]  # moov whole, mdat only started
    info = media.mp4_info(head, len(data))
    assert info is not None and info.faststart and (info.width, info.height) == (1080, 1920)


def test_mp4_info_with_moov_beyond_the_header_is_none() -> None:
    data = mp4_bytes(faststart=False)
    assert media.mp4_info(data[: data.index(b"moov") - 4 - 10], len(data)) is None


def test_mp4_info_of_a_zero_timescale_is_none() -> None:
    assert media.mp4_info(mp4_bytes(timescale=0)) is None


def test_find_ffmpeg_needs_apple_silicon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _fake_package(monkeypatch, tmp_path)
    assert media.find_ffmpeg("linux", "arm64") is None
    assert media.find_ffmpeg("darwin", "x86_64") is None
    assert media.find_ffmpeg("darwin", "arm64") == binary


def test_find_ffmpeg_never_imports_the_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_package(monkeypatch, tmp_path)
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    media.find_ffmpeg("darwin", "arm64")
    assert "imageio_ffmpeg" not in sys.modules


def test_find_ffmpeg_is_none_without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.importlib.util, "find_spec", lambda name: None)
    assert media.find_ffmpeg("darwin", "arm64") is None


@pytest.mark.parametrize("error", [ValueError("spec is None"), ImportError("broken parent")])
def test_find_ffmpeg_survives_find_spec_errors(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def boom(name: str) -> None:
        raise error

    monkeypatch.setattr(media.importlib.util, "find_spec", boom)
    assert media.find_ffmpeg("darwin", "arm64") is None


def test_find_ffmpeg_needs_an_executable_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _fake_package(monkeypatch, tmp_path)
    binary.chmod(0o644)
    assert media.find_ffmpeg("darwin", "arm64") is None
    (binary.parent / "ffmpeg-other").write_text("x")
    (binary.parent / "ffmpeg-other").chmod(0o755)
    assert media.find_ffmpeg("darwin", "arm64") == binary.parent / "ffmpeg-other"


def _fake_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    package = tmp_path / "imageio_ffmpeg"
    (package / "binaries").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    binary = package / "binaries" / "ffmpeg-macos-aarch64-v9.0"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    spec = types.SimpleNamespace(origin=str(package / "__init__.py"), submodule_search_locations=[str(package)])
    monkeypatch.setattr(media.importlib.util, "find_spec", lambda name: spec if name == "imageio_ffmpeg" else None)
    return binary
