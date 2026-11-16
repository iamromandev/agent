"""NFR3: what the guardrails add per command. The sandbox start-up test needs macOS."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
from src.engine.guardrails import sandbox as sandbox_module
from src.engine.guardrails.config import GuardConfig
from src.engine.guardrails.paths import ProtectedPaths
from src.engine.guardrails.redact import Redactor
from src.engine.guardrails.sandbox import SANDBOX_EXEC
from src.engine.guardrails.setup import prepare

SANDBOX_START_LIMIT_MS = 50.0  # NFR3: added per bash command, median
RUNS = 25


def _median_ms(argv: list[str]) -> float:
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


@pytest.mark.skipif(sys.platform != "darwin" or not SANDBOX_EXEC.exists(), reason="needs macOS and /usr/bin/sandbox-exec")
def test_the_sandbox_adds_at_most_50_ms_to_the_median_command_start(tmp_path: Path) -> None:
    home = (tmp_path / "home").resolve()
    home.mkdir()
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    guard = prepare(workspace, None, None, {}, "Darwin", home=home)
    assert guard.sandbox.profile is not None, guard.warnings
    plain = ["/usr/bin/true"]
    _median_ms(plain)  # warm the file cache

    bare = _median_ms(plain)
    wrapped = _median_ms(guard.sandbox.wrap(plain))

    print(f"NFR3 sandbox start: bare={bare:.1f}ms sandboxed={wrapped:.1f}ms added={wrapped - bare:.1f}ms")
    assert wrapped - bare <= SANDBOX_START_LIMIT_MS


def test_building_the_profile_with_many_settings_patterns_is_fast(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    patterns = tuple(f"secrets{index}/**" for index in range(100)) + tuple(f"*.tok{index}" for index in range(100))
    config = GuardConfig(extra_protected=patterns)
    start = time.perf_counter()
    sandbox_module.build_profile(
        workspace, [workspace], [], ProtectedPaths(home, config.extra_protected), True
    )
    elapsed = time.perf_counter() - start

    print(f"profile with 200 patterns: {elapsed * 1000:.1f}ms")
    assert elapsed < 0.5


REDACT_LIMIT_MS = 5.0  # NFR3: per 100 KB of output
REDACT_RUNS = 7


def _best_ms(redactor: Redactor, text: str) -> float:
    samples = []
    for _ in range(REDACT_RUNS):
        start = time.perf_counter()
        redactor.redact(text)
        samples.append((time.perf_counter() - start) * 1000)
    return min(samples)  # the least disturbed run: a loaded CI machine must not fail the limit


def _output_100kb(with_secrets: bool) -> str:
    lines = [
        "src/engine/loop.py:123:    result = await self._approve_and_run(tool, call, prepared, mode, emit)",
        "commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904 id 123e4567-e89b-12d3-a456-426614174000",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-sk-eyJ",
        "the password policy requires a long passphrase; see sk-short and ghp_short",
    ]
    secrets = [
        "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "password=hunter2 key AKIAIOSFODNN7EXAMPLE",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\n-----END PRIVATE KEY-----",
    ]
    out: list[str] = []
    size = 0
    index = 0
    while size < 100_000:
        line = lines[index % len(lines)]
        if with_secrets and index % 50 == 0:
            line = secrets[(index // 50) % len(secrets)]
        out.append(line)
        size += len(line) + 1
        index += 1
    return "\n".join(out)


@pytest.mark.parametrize("with_secrets", [False, True])
def test_redacting_100_kb_takes_at_most_5_ms(with_secrets: bool) -> None:
    redactor = Redactor({f"VAR_{i}_TOKEN": f"value-number-{i:04d}-secret" for i in range(20)})
    text = _output_100kb(with_secrets)
    assert len(text) >= 100_000
    redactor.redact(text)  # warm up

    elapsed = _best_ms(redactor, text)

    print(f"NFR3 redact 100KB (secrets={with_secrets}): {elapsed:.2f}ms")
    assert elapsed <= REDACT_LIMIT_MS


# Worst cases for every pattern, 300 KB each: the time must stay linear (a quadratic pattern took 15 s on `eyJ` * 80k).
ADVERSARIAL_LIMIT_S = 0.25
ADVERSARIAL_SIZE = 300_000
_B64 = "A" * 40


def _repeat(unit: str) -> str:
    return unit * (ADVERSARIAL_SIZE // len(unit))


_DEEP_OPEN = "{" + "\\" * 3 + '"password' + "\\" * 3 + '": ' + "\\" * 3 + '"'  # JSON in JSON in JSON
_DEEP_INNER = "a" + "\\" * 7 + '"'  # an escaped quote of the value at that depth

ADVERSARIAL = {
    "jwt-header-run": _repeat("eyJ"),
    "jwt-dots": _repeat("eyJa."),
    "jwt-segments": _repeat("eyJa.b"),
    "jwt-long-header": "eyJ" + "a" * ADVERSARIAL_SIZE,
    "jwt-after-letter": _repeat("xeyJabc"),
    "github-pat": _repeat("github_pat_"),
    "github-pat-short": _repeat("github_pat_" + "a" * 21 + " "),
    "github": _repeat("ghp_"),
    "github-short": _repeat("ghp_" + "a" * 35 + " "),
    "github-gh": _repeat("gh"),
    "aws-key": _repeat("AKIA"),
    "aws-key-near": _repeat("AKIAAAAAAAAAAAAAAAAAAA"),
    "slack": _repeat("xox"),
    "slack-short": _repeat("xoxb-" + "a" * 9 + " "),
    "sk": _repeat("sk-"),
    "sk-short": _repeat("sk-" + "a" * 19 + " "),
    "sk-after-letter": _repeat("ask-"),
    "sk-ant": _repeat("sk-ant-"),
    "sk-ant-after-letter": _repeat("ask-ant-"),
    "pem-begin": _repeat("-----BEGIN PRIVATE KEY-----"),
    "pem-begin-label": _repeat("-----BEGIN "),
    "pem-begin-long-label": "-----BEGIN " + "A" * ADVERSARIAL_SIZE,
    "pem-end": _repeat("-----END PRIVATE KEY-----\n"),
    "pem-end-blank": _repeat("-----END PRIVATE KEY-----\n\n"),
    "pem-end-label": _repeat("-----END "),
    "pem-end-long-label": "-----END " + "A" * ADVERSARIAL_SIZE,
    "aws-secret": _repeat("aws_secret_access_key="),
    "aws-secret-quoted": _repeat('aws_secret_access_key="'),
    "password": _repeat("password="),
    "password-spaces": _repeat("password" + " " * 20),
    "password-dq-escapes": 'password="' + '\\"' * (ADVERSARIAL_SIZE // 2),
    "password-dq-open": _repeat('password="a'),
    "password-escaped-json": '{\\"password\\": \\"' + "a\\" * (ADVERSARIAL_SIZE // 2),
    "password-sq": _repeat("password='a"),
    "password-ident": "a" * ADVERSARIAL_SIZE + "password=a",
    "password-call": _repeat("password=getpass("),
    "redacted-placeholders": _repeat("password=[redacted:password]"),
    # An env secret is set in every one of these: its pass skips placeholders and once scanned to the end of the text
    # from each `[redacted:` (13 s for 200 KB).
    "redacted-open": _repeat("[redacted:"),
    "redacted-open-names": _repeat("[redacted:env:A_LONG_NAME_"),
    "redacted-long-name": _repeat("[redacted:" + "a" * 190),
    "redacted-open-newline": _repeat("[redacted:\n"),
    # Context checks of the password pass: none may read back or ahead from every key.
    "password-closing-quote": "x'" + _repeat("password: 'a"),
    "password-closing-quote-dq": 'x"' + _repeat('password: "a'),
    "password-quote-lines": _repeat("'password: 'a\n"),
    "password-reference": _repeat("password=$HOME "),
    "password-braced": _repeat("password=${a "),
    "password-call-args": _repeat("password=a(" + "b" * 100),
    "password-dotted": _repeat("password=a.b.c.d.e.f "),
    "password-structure": _repeat("password={"),
    "password-backslashes": 'password="' + "\\" * ADVERSARIAL_SIZE,
    "password-backslash-quotes": 'password="' + '\\\\"' * (ADVERSARIAL_SIZE // 3),
    "password-deep-escapes": _DEEP_OPEN + _DEEP_INNER * ((ADVERSARIAL_SIZE - len(_DEEP_OPEN)) // len(_DEEP_INNER)),
    "password-key-backslashes": "password" + "\\" * ADVERSARIAL_SIZE,
    "orphan-end-long-line": _repeat("A" * 180 + "\n-----END PRIVATE KEY-----\n"),
    "orphan-end-prose": _repeat("some prose here\n-----END PRIVATE KEY-----\n"),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_an_adversarial_text_is_redacted_in_linear_time(name: str) -> None:
    redactor = Redactor({f"VAR_{i}_TOKEN": f"value-number-{i:04d}-secret" for i in range(20)})
    text = ADVERSARIAL[name]
    assert len(text) >= ADVERSARIAL_SIZE - 100

    elapsed = _best_ms(redactor, text) / 1000

    print(f"adversarial {name}: {elapsed * 1000:.1f}ms for {len(text)} chars")
    assert elapsed <= ADVERSARIAL_LIMIT_S


def test_a_text_full_of_overlapping_env_values_is_redacted_in_linear_time() -> None:
    redactor = Redactor({"A_TOKEN": "abcdefgh12345678", "B_TOKEN": "12345678zzzzzzzz", "C_TOKEN": "abcdefgh"})
    text = "abcdefgh12345678zzzzzzzz abcdefgh " * (ADVERSARIAL_SIZE // 34)

    elapsed = _best_ms(redactor, text) / 1000

    print(f"adversarial env values: {elapsed * 1000:.1f}ms for {len(text)} chars")
    assert elapsed <= ADVERSARIAL_LIMIT_S
