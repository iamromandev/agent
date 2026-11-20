"""The macOS sandbox for spawned processes (R11-R13): an SBPL profile passed inline to `sandbox-exec -p`.

Pure text plus one probe at launch; nothing here starts a process except `default_probe`. SBPL lets later rules win,
so the profile is ordered: allow everything, deny writes, allow the writable roots, deny writes to the agent's own
files and to protected paths (and their removal or rename), then the read denials with the system folders re-allowed
before the path denials. Case folding in the name and glob regexes is ASCII only. See the plan's spike results.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.engine.guardrails.config import GuardConfig
from src.engine.guardrails.paths import FIRMLINK, ProtectedPaths
from src.engine.text import escape

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

TEMP_ROOTS = ("/tmp", "/private/tmp", "/private/var/folders")
CACHES = (".cache", "Library/Caches", ".npm", ".bun", ".cargo/registry", ".local/share/uv", ".local/state")
# `/dev/ttysNNN` are the pty slaves (a terminal tab's own pty is allocated after the profile is built, so its name
# cannot be listed); `/dev/tty.*` (serial and Bluetooth ports) stay closed.
DEVICE_WRITES = (
    '(literal "/dev/null")',
    '(literal "/dev/stdout")',
    '(literal "/dev/stderr")',
    '(literal "/dev/ptmx")',
    '(literal "/dev/tty")',
    '(regex #"^/dev/fd/")',
    '(regex #"^/dev/ttys[0-9]+$")',
    '(regex #"^/dev/pty")',
)
# Read again after the name rules (`*.pem`, `*.key`): the system's CA bundles and the one `certifi` ships in a venv.
SYSTEM_READS = ("/etc", "/private/etc", "/usr", "/opt/homebrew", "/System", "/Library")
CERTIFI_BUNDLE = r'(regex #"/certifi/cacert\.pem$")'
PROBE_TIMEOUT = 10.0
# Not a network jail: DNS still resolves, Unix sockets are open and the machine's own LAN address is not "localhost".
NETWORK_LOCAL_ONLY = '(deny network*)(allow network* (remote ip "localhost:*") (remote unix-socket))'
NOT_MACOS = "unavailable: not macOS"
NO_EXEC = "unavailable: sandbox-exec not found"
PROBE_FAILED = "unavailable: sandbox-exec could not start a test command"
NOT_BUILT = "unavailable: the profile could not be built"


@dataclass(frozen=True)
class Sandbox:
    profile: str | None  # None: unsandboxed
    reason: str  # "on (macOS sandbox-exec)", "off: disabled in settings", ...
    network: bool
    writable: tuple[Path, ...]

    def wrap(self, argv: Sequence[str]) -> list[str]:
        """`argv` as it must be started: behind `sandbox-exec` with the profile, or unchanged without one."""
        if self.profile is None:
            return list(argv)
        return [str(SANDBOX_EXEC), "-p", self.profile, *argv]


def _string(text: str) -> str:
    """An SBPL string literal: `\\` and `"` escaped."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


_REGEX_SPECIAL = frozenset("\\.^$*+?()[]{}|")


def _regex_char(char: str, ignore_case: bool = True) -> str:
    """One literal character in an SBPL regex. A `"` cannot be written inside `#"..."` (a `\\"` ends the string),
    so it becomes `.`, which only widens a denial."""
    if char == '"':
        return "."
    if ignore_case and char.isascii() and char.isalpha():
        return f"[{char.lower()}{char.upper()}]"
    return "\\" + char if char in _REGEX_SPECIAL else char


def _bracket(pattern: str, start: int) -> tuple[str, int] | None:
    """A glob `[...]` starting at `start` as a regex bracket expression, and the index after it; None if unclosed."""
    index = start + 1
    negate = pattern[index : index + 1] in ("!", "^")
    if negate:
        index += 1
    first = index
    if pattern[index : index + 1] == "]":
        index += 1
    end = pattern.find("]", index)
    if end < 0:
        return None
    # Not expressible in a bracket of a `#"..."` string: dropped (the match gets narrower).
    chars = [char for char in pattern[first:end] if char not in '"\\[']
    members: list[str] = []
    at = 0
    while at < len(chars):
        low = chars[at]
        if at + 2 < len(chars) and chars[at + 1] == "-":  # a range: its other-case twin is added, not each letter
            high = chars[at + 2]
            members.append(f"{low}-{high}")
            if _ascii_letters(low, high) and low.islower() == high.islower():
                members.append(f"{low.swapcase()}-{high.swapcase()}")
            at += 3
        else:
            members.append(f"{low}{low.swapcase()}" if _ascii_letters(low) else low)
            at += 1
    return "[" + ("^" if negate else "") + "".join(members) + "]", end + 1


def _ascii_letters(*chars: str) -> bool:
    return all(char.isascii() and char.isalpha() for char in chars)


def glob_regex(pattern: str) -> str:
    """The glob as an SBPL regex body (no anchors), ignoring case like `ProtectedPaths`. `*` and `?` stay inside a
    path segment; `**/` is any folders (maybe none) and a trailing `/**` or `**` is anything."""
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**", index):
            if pattern[index + 2 : index + 3] == "/":
                out.append("(.*/)?")
                index += 3
            else:
                out.append(".*")
                index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        elif char == "[" and (found := _bracket(pattern, index)) is not None:
            out.append(found[0])
            index = found[1]
        else:
            out.append(_regex_char(char))
            index += 1
    return "".join(out)


def _name_regex(pattern: str) -> str:
    return f'#"/{glob_regex(pattern)}$"'


def _path_regexes(pattern: str) -> list[str]:
    """The regexes for a path glob: anchored, and for an absolute one also behind the macOS firmlink prefix (an
    optional group in front does not match reliably in SBPL regexes, so it is a second regex)."""
    regexes = [f'#"^{glob_regex(pattern)}$"']
    if pattern.startswith("/") and not pattern.startswith(FIRMLINK):
        regexes.append(f'#"^{FIRMLINK}{glob_regex(pattern)}$"')
    return regexes


def _variants(path: Path) -> list[str]:
    """Every spelling a rule for `path` is written with: as given, the real path when it differs, and each with the
    macOS firmlink prefix (SBPL may match the path it is given)."""
    spellings: list[str] = []
    for candidate in (path, path.resolve()):
        text = str(candidate)
        if text not in spellings:
            spellings.append(text)
    for text in list(spellings):
        if text.startswith("/") and not text.startswith(FIRMLINK):
            spellings.append(FIRMLINK + text)
    return spellings


def _unique(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path not in found:
            found.append(path)
    return found


def build_profile(
    workspace: Path,
    writable: Sequence[Path],
    deny_write: Sequence[Path],
    deny_read: ProtectedPaths,
    network: bool,
) -> str:
    """The SBPL profile text. `writable` and `deny_write` are folders or files (`subpath` matches both); the
    workspace's own entry and its `.agent` folder are also denied removal and rename as literals, so neither can be
    moved away and back (the protected files in it would then be writable under another path)."""
    workspace = workspace.resolve()
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    roots = _unique(path.resolve() for path in writable)
    allowed = " ".join([*(f"(subpath {_string(str(root))})" for root in roots), *DEVICE_WRITES])
    lines.append(f"(allow file-write* {allowed})")

    denied = [f"(subpath {_string(text)})" for path in deny_write for text in _variants(path)]
    if denied:
        lines.append(f"(deny file-write* {' '.join(dict.fromkeys(denied))})")
    # Only removal and rename, not writes: `touch .`, `tar -x` and `cp -R x/. .` must work in the workspace.
    literals = [f"(literal {_string(text)})" for path in (workspace, workspace / ".agent") for text in _variants(path)]
    lines.append(f"(deny file-write-unlink {' '.join(dict.fromkeys(literals))})")

    builtin = [_name_regex(name) for name in deny_read.builtin_name_patterns]
    exempt = [_name_regex(name) for name in sorted(deny_read.exempt_names)]
    extra = [_name_regex(name) for name in deny_read.extra_name_patterns]
    globs = [regex for pattern in deny_read.glob_patterns() for regex in _path_regexes(pattern)]
    # Protected files stay unwritable even inside a writable root, and cannot be renamed or removed (a rename would
    # leave the contents where no pattern protects them): `file-write*` includes the unlink a rename needs.
    names = f"(regex {' '.join(builtin)})"
    lines.append(f"(deny file-write* (require-all {names} (require-not (regex {' '.join(exempt)}))))")
    for root in deny_read.home_roots():
        lines.extend(f"(deny file-write* (subpath {_string(text)}))" for text in _variants(root))
    if extra or globs:
        lines.append(f"(deny file-write* (regex {' '.join([*extra, *globs])}))")
    # The folder a `secrets/**` pattern names is not matched by it, but renaming it away would hide what is inside.
    folders = [
        regex for pattern in deny_read.glob_patterns() if pattern.endswith("/**") for regex in _path_regexes(pattern[:-3])
    ]
    if folders:
        lines.append(f"(deny file-write-unlink (regex {' '.join(folders)}))")

    lines.append(f"(deny file-read* (regex {' '.join(builtin)}))")
    lines.append(f"(allow file-read* (regex {' '.join(exempt)}))")
    if extra:
        lines.append(f"(deny file-read* (regex {' '.join(extra)}))")
    system = " ".join([*(f"(subpath {_string(folder)})" for folder in SYSTEM_READS), CERTIFI_BUNDLE])
    lines.append(f"(allow file-read* {system})")

    for root in deny_read.home_roots():
        lines.extend(f"(deny file-read* (subpath {_string(text)}))" for text in _variants(root))
    if globs:
        lines.append(f"(deny file-read* (regex {' '.join(globs)}))")
    if not network:
        lines.append(NETWORK_LOCAL_ONLY)
    return "\n".join(lines) + "\n"


def default_probe(argv: list[str]) -> bool:
    """Run `argv` (the sandboxed `/usr/bin/true`) once; True when it exits 0."""
    try:
        done = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=PROBE_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _xdg(env: Mapping[str, str], name: str, fallback: Path) -> Path:
    value = env.get(name)
    return Path(value) if value and Path(value).is_absolute() else fallback


def agent_folders(env: Mapping[str, str], home: Path, sessions: Path | None = None) -> tuple[Path, ...]:
    """The agent's own folders: `<config>/agent` (settings, MCP servers, commands, skills), `<data>/agent` (trust
    stores, checkpoints) and the sessions folder when given. No command may write them (the profile denies it) and
    accept-edits mode never approves a write into them."""
    config = _xdg(env, "XDG_CONFIG_HOME", home / ".config") / "agent"
    data = _xdg(env, "XDG_DATA_HOME", home / ".local" / "share") / "agent"
    return (config, data) if sessions is None else (config, data, sessions)


def _extra_writable(entries: Sequence[str], home: Path) -> tuple[list[Path], list[str]]:
    """The global `sandbox.write` entries as paths (`~/x` and absolute ones), and a warning for each other one."""
    paths: list[Path] = []
    warnings: list[str] = []
    for entry in entries:
        text = entry.strip()
        if text.startswith("~/"):
            paths.append(home / text[2:])
        elif text.startswith("/"):
            paths.append(Path(text))
        else:
            warnings.append(
                f'guardrails.sandbox.write entry "{escape(entry)}" must be an absolute path or start with "~/"; ignored'
            )
    return paths, warnings


def _covers(path: Path, folder: Path) -> bool:
    """True when `path` is `folder` or one of its ancestors (both resolved)."""
    path, folder = path.resolve(), folder.resolve()
    return path == folder or path in folder.parents


def _workspace_warnings(workspace: Path, paths: ProtectedPaths) -> list[str]:
    """A warning when the workspace is, or contains, the home folder or a protected folder: the profile then makes
    the workspace writable and still denies those places, so commands cannot use them at all."""
    inside = [root for root in paths.home_roots() if _covers(workspace, root)]
    if not inside:
        return []
    shown = ", ".join(escape(str(root)) for root in inside[:3]) + (f" and {len(inside) - 3} more" if len(inside) > 3 else "")
    return [f"sandbox: the workspace {escape(str(workspace))} contains protected paths ({shown}); commands cannot read or write them"]


def make_sandbox(
    cfg: GuardConfig,
    workspace: Path,
    home: Path,
    env: Mapping[str, str],
    system: str,
    probe: Callable[[list[str]], bool],
    *,
    sessions: Path | None = None,
    protected: ProtectedPaths | None = None,
) -> tuple[Sandbox, str | None]:
    """The launch's sandbox, and a warning (one string, lines joined) when it is not what the settings asked for.

    Unsandboxed, with the reason, when it is disabled, the system is not macOS, `sandbox-exec` is missing or the
    probe (the real profile around `/usr/bin/true`) fails: a broken sandbox never stops the app (NFR4, R13)."""

    def unsandboxed(reason: str, warning: str | None) -> tuple[Sandbox, str | None]:
        return Sandbox(None, reason, cfg.network, ()), warning

    if not cfg.sandbox_enabled:
        return unsandboxed("off: disabled in settings", None)
    if system != "Darwin":
        return unsandboxed(NOT_MACOS, "sandbox unavailable: not macOS; commands run without a sandbox")
    if not SANDBOX_EXEC.exists():
        return unsandboxed(NO_EXEC, f"sandbox unavailable: sandbox-exec not found at {escape(str(SANDBOX_EXEC))}; commands run without a sandbox")
    extra, warnings = _extra_writable(cfg.extra_write, home)
    temp = env.get("TMPDIR")
    temp_root = Path(temp) if temp and Path(temp).is_absolute() else None
    if temp_root is not None and _covers(temp_root, home):
        warnings.append(f'TMPDIR "{escape(temp or "")}" is the root, the home folder or above it; it is not made writable')
        temp_root = None
    try:
        writable = _unique(
            path.resolve()
            for path in (
                workspace,
                *([temp_root] if temp_root is not None else []),
                *(Path(root) for root in TEMP_ROOTS),
                *(home / cache for cache in CACHES),
                *extra,
            )
        )
        deny_write = list(agent_folders(env, home, sessions))
        agent = workspace / ".agent"
        deny_write += [agent / "settings.json", agent / "mcp.json", agent / "commands", agent / "skills", agent / "agents"]
        paths = protected if protected is not None else ProtectedPaths(home, cfg.extra_protected)
        warnings.extend(_workspace_warnings(workspace, paths))
        profile = build_profile(workspace, writable, deny_write, paths, cfg.network)
    except Exception as exc:  # NFR4: an odd path or pattern must not stop the launch
        reason = f"sandbox unavailable: could not build the profile ({escape(str(exc))}); commands run without a sandbox"
        return unsandboxed(NOT_BUILT, reason)
    try:
        working = probe([str(SANDBOX_EXEC), "-p", profile, "/usr/bin/true"])
    except Exception:
        working = False
    if not working:
        return unsandboxed(
            PROBE_FAILED, "sandbox unavailable: sandbox-exec could not start a test command; commands run without a sandbox"
        )
    warning = "\n".join(warnings) if warnings else None
    return Sandbox(profile, "on (macOS sandbox-exec)", cfg.network, tuple(writable)), warning
