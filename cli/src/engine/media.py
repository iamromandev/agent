"""Media tools' shared facts: the speech engine, the voices, and reading audio headers.

The engine (mlx-audio, misaki) is an optional extra and is never imported here: it runs in a helper process.
UI-free: no `textual`, no `httpx`.
"""

from __future__ import annotations

import importlib.util
import platform as platform_module
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
DEFAULT_VOICE = "af_heart"
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 0.5, 2.0, 1.0
MAX_TEXT = 4000
MAX_AUDIO_BYTES = 50 * 1024 * 1024
TEXT_NAME, WAV_NAME, SKIPPED_NAME = "text.txt", "speech.wav", "skipped.json"
_MAX_CHUNKS = 64  # RIFF chunks looked at before giving up: bounds the walk over a hostile file

_SPEC = importlib.util.find_spec("src.engine.speech_helper")  # found, never imported: it needs the media extra
SPEECH_HELPER: Path | None = Path(_SPEC.origin) if _SPEC is not None and _SPEC.origin else None


def find_speech(system: str = sys.platform, machine: str | None = None) -> Path | None:
    """The speech helper script when the engine is installed (Apple Silicon macOS, mlx-audio and misaki importable)."""
    machine = platform_module.machine() if machine is None else machine
    if system != "darwin" or machine != "arm64":
        return None
    try:
        if importlib.util.find_spec("mlx_audio") is None or importlib.util.find_spec("misaki") is None:
            return None
    except (ValueError, ImportError):  # a module with `__spec__ = None`, or a broken parent package
        return None
    helper = SPEECH_HELPER
    return helper if helper is not None and helper.is_file() else None


def sniff_audio(head: bytes) -> str | None:
    """`wav`, `mp3` or `m4a` from the first bytes of a file, or None. The name is never used."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and (head[1] >> 3) & 3 != 1 and (head[1] >> 1) & 3 != 0:
        return "mp3"  # a frame sync with neither the reserved MPEG version (01) nor the reserved layer (00)
    if head[4:8] == b"ftyp" and head[8:12] in (b"M4A ", b"mp42", b"isom"):
        return "m4a"
    return None


@dataclass(frozen=True)
class WavInfo:
    rate: int
    channels: int
    bits: int
    seconds: float


_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


def wav_info(data: bytes) -> WavInfo | None:
    """Sample rate, channels, bits and length of a PCM (16/24/32 bit) or float32 WAV, or None when it is malformed,
    of another format, or holds no audio. The length comes from the data actually present."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    fmt: tuple[int, int, int, int, int] | None = None  # code, channels, rate, byte rate, bits
    for _ in range(_MAX_CHUNKS):
        if pos + 8 > len(data):
            return None
        name = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if name == b"fmt ":
            if size < 16 or body + 16 > len(data):
                return None
            code, channels, rate, byte_rate, _align, bits = struct.unpack_from("<HHIIHH", data, body)
            if code == _EXTENSIBLE:
                if size < 40 or body + 26 > len(data):
                    return None
                code = struct.unpack_from("<H", data, body + 24)[0]
            fmt = (code, channels, rate, byte_rate, bits)
        elif name == b"data":
            if fmt is None:
                return None
            code, channels, rate, byte_rate, bits = fmt
            ok = (code == _PCM and bits in (16, 24, 32)) or (code == _FLOAT and bits == 32)
            if not ok or channels < 1 or rate < 1:
                return None
            width = channels * bits // 8
            present = min(size, len(data) - body)
            seconds = (present - present % width) / (rate * width)
            return WavInfo(rate, channels, bits, seconds) if seconds > 0 else None
        pos = body + size + (size & 1)
    return None
