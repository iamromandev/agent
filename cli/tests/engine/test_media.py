"""Media helpers: sniffing audio, reading WAV headers, finding the speech engine."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from src.engine import media
from src.engine.media import DEFAULT_VOICE, VOICES, WavInfo, find_speech, sniff_audio, wav_info


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
