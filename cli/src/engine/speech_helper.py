"""Speech helper for `speak`. A script, run as `python -I speech_helper.py ...`; never imported by the agent.

    --model M --voice V --speed S --text-file text.txt --out speech.wav --skipped skipped.json

It runs in its own process (filtered environment) because it loads mlx-audio (Kokoro), misaki, torch and
spacy, which the agent never imports, and reads the text only from a file. It never runs pip and imports nothing from
the agent.

espeak is broken on macOS (every espeakng-loader release ignores its data path and calls C `exit()`, which kills the
process silently), so a stub `misaki.espeak` is installed before anything imports misaki: its fallback fails to
construct, mlx-audio then runs without one, and out-of-dictionary words are dropped. The words dropped are written to
`--skipped` as a JSON list so the caller can tell the model to respell them.

Exit codes: 0 ok, 1 other failure (one short stderr line), 2 usage, 3 the model could not be loaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from collections.abc import Iterable
from typing import Any

EXIT_MODEL = 3
UNKNOWN = "❓"  # misaki's mark for a word it cannot pronounce


def install_espeak_stub() -> None:
    """Replace `misaki.espeak` with a module whose fallback cannot be built."""
    stub: Any = types.ModuleType("misaki.espeak")

    class EspeakFallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("espeak is disabled")

    stub.EspeakFallback = EspeakFallback
    sys.modules["misaki.espeak"] = stub


def skipped_words(tokens: Iterable[Any], unknown: str = UNKNOWN) -> list[str]:
    """The words of `tokens` (objects with `text` and `phonemes`) that have no pronunciation (phonemes None) or only a
    partial one (the unknown mark in it), once each, in order. Empty phonemes mean a silent symbol such as `$`, and
    whitespace-only tokens are not words."""
    found: list[str] = []
    for token in tokens:
        text = token.text
        phonemes = token.phonemes
        if not text.strip() or not (phonemes is None or unknown in phonemes):
            continue
        if text not in found:
            found.append(text)
    return found


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech_helper")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--speed", required=True, type=float)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skipped", required=True)
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    with open(args.text_file, encoding="utf-8") as handle:
        text = handle.read()
    install_espeak_stub()
    try:
        load_model = importlib.import_module("mlx_audio.tts.utils").load_model
        model = load_model(model_path=args.model)
    except Exception as exc:
        print(f"could not load the model {args.model}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_MODEL
    np = importlib.import_module("numpy")
    lang = args.voice[0]
    parts = [
        np.array(result.audio)
        for result in model.generate(text=text, voice=args.voice, speed=args.speed, lang_code=lang)
    ]
    if not parts:
        print("the model produced no audio", file=sys.stderr)
        return 1
    audio = np.concatenate(parts)
    write = importlib.import_module("mlx_audio.audio_io").write
    write(args.out, audio, model.sample_rate)
    write_skipped(args.skipped, text, lang)
    return 0


def write_skipped(path: str, text: str, lang: str) -> None:
    """Write the words misaki cannot pronounce to `path`. Never raises: finished audio must not be lost to this step,
    so on any failure it writes `[]` (when it can) and prints one warning line."""
    try:
        en = importlib.import_module("misaki.en")
        g2p = en.G2P(trf=False, british=lang == "b", fallback=None, unk=UNKNOWN)
        _, tokens = g2p(text)
        words = skipped_words(tokens, UNKNOWN)
    except Exception as exc:
        print(f"could not find the skipped words: {type(exc).__name__}: {exc}", file=sys.stderr)
        words = []
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(words, handle, ensure_ascii=False)
    except Exception as exc:
        print(f"could not write the skipped words: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
