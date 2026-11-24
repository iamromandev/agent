"""Protected paths: credentials the tools must not touch without the user's say (R3). Pure matching, no I/O."""

from __future__ import annotations

import fnmatch
import os
import pwd
import stat
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

BUILTIN_HOME = (
    "~/.ssh/**",
    "~/.aws/**",
    "~/.gnupg/**",
    "~/.config/gh/**",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.kube/config",
    "~/Library/Keychains/**",
)
BUILTIN_NAMES = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", "id_ecdsa*")
NAME_EXCEPTIONS = (".env.example", ".env.sample", ".env.template")  # of the built-in names only

_WILDCARDS = frozenset("*?[")

FIRMLINK = "/System/Volumes/Data"  # macOS: `/Users/me/x` and `/System/Volumes/Data/Users/me/x` are one file
FIRMLINK_LOWER = FIRMLINK.lower()
STRIP_FIRMLINK = sys.platform == "darwin"
MAX_TRACKED_FILES = 2_000  # regular files under protected folders whose identity is recorded at launch (hard links)
MAX_TRACKED_DEPTH = 4  # folder levels below a protected folder that are scanned for them
MAX_WORKSPACE_DEPTH = 12  # the same, for the workspace scan for files with a protected name
MAX_TRACKED_DIRS = 5_000  # folders visited by one scan (a protected folder, or the workspace)
SCAN_SKIP = frozenset({".git", "node_modules", ".venv", "__pycache__"})  # the workspace scan does not enter these

type FileId = tuple[int, int]  # (st_dev, st_ino)


def strip_firmlink(text: str) -> str:
    """`text` without a leading macOS `/System/Volumes/Data` (`Path.resolve()` keeps it); unchanged elsewhere."""
    # APFS is case-insensitive: `/system/volumes/data/...` is the same place, so the prefix is matched without case
    if STRIP_FIRMLINK and text[: len(FIRMLINK)].lower() == FIRMLINK_LOWER and text[len(FIRMLINK) :][:1] in ("", "/"):
        return text[len(FIRMLINK) :] or "/"
    return text


def _canonical(text: str) -> str:
    """The spelling patterns are matched in: no macOS firmlink prefix, Unicode NFC (APFS and HFS+ treat the
    decomposed and composed forms of a name as one)."""
    return unicodedata.normalize("NFC", strip_firmlink(text))


def login_home() -> Path | None:
    """The home folder the account database names, when it differs from `Path.home()` (HOME can be set to anything)."""
    try:
        recorded = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        return None
    if not recorded.is_absolute():
        return None
    current = Path.home()
    return None if recorded == current or recorded.resolve() == current.resolve() else recorded


@dataclass
class _Budget:
    """What one scan may still visit."""

    files: int = 0
    dirs: int = 0


def fd_path(fd: int) -> str | None:
    """Where the open file `fd` really is, as the OS reports it (symlinks resolved); None when it cannot say."""
    try:
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
            return os.fsdecode(raw.split(b"\0", 1)[0]) or None
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ImportError, AttributeError):
        return None


def _identity(path: Path | str) -> tuple[FileId, int] | None:
    """`((st_dev, st_ino), st_mode)` following symlinks; None when it does not exist or cannot be examined."""
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return None
    return (info.st_dev, info.st_ino), info.st_mode


class ProtectedPaths:
    """A pattern with a `/` is a path (`~` is the home folder; a relative one matches under any folder); one
    without matches the file name anywhere. Matching ignores case (the file system may). `is_protected` takes a
    path that is already resolved (symlinks, `~`, `..`): resolving is the caller's job.

    A path also counts when the file system says it is a protected file or lies inside a protected folder, by
    `(st_dev, st_ino)` recorded at launch: another spelling of the same place (macOS firmlinks, a symlink that
    was not resolved) and hard links to files that existed then are caught without a pattern for them."""

    def __init__(
        self, home: Path, extra: Sequence[str], workspace: Path | None = None, *, also_home: Path | None = None
    ) -> None:
        self._extra = tuple(text for text in (pattern.strip() for pattern in extra) if text)
        self.home_folders = [home] if also_home is None else [home, also_home]  # as given, for `/guardrails`
        homes: list[Path] = []
        for folder in self.home_folders:
            homes.extend([folder] if folder.resolve() == folder else [folder.resolve(), folder])
        self._homes = [_canonical(str(h)) for h in homes]
        self._path_patterns: list[str] = []
        self.builtin_name_patterns = [name.lower() for name in BUILTIN_NAMES]  # lower case, matched against the lower-cased name
        self.extra_name_patterns: list[str] = []
        self.exempt_names = frozenset(name.lower() for name in NAME_EXCEPTIONS)
        for pattern in (*BUILTIN_HOME, *self._extra):
            self._add(_canonical(pattern))
        self._file_ids: set[FileId] = set()
        self._folder_ids: set[FileId] = set()
        self.tracking_truncated = False  # some files under a protected folder or in the workspace were beyond the bounds
        self._track()
        if workspace is not None:
            self._track_named(workspace)

    def _add(self, pattern: str) -> None:
        pattern = pattern.strip()
        if not pattern:
            return
        if "/" not in pattern:
            self.extra_name_patterns.append(pattern.lower())
            return
        if ".." in pattern.split("/"):  # `config.py` warns; here it is never matched
            return
        base = pattern.rstrip("/")
        if not base:
            return
        # A pattern that names a folder (a trailing `/`, or no wildcard in the last segment) covers what is inside.
        names_folder = pattern.endswith("/") or not _WILDCARDS.intersection(base.rsplit("/", 1)[-1])
        for variant in (base, f"{base}/**") if names_folder else (pattern,):
            self._add_path(variant)

    def _add_path(self, pattern: str) -> None:
        if pattern.startswith("~/"):
            self._path_patterns.extend(f"{home}{pattern[1:]}" for home in self._homes)
        elif pattern.startswith("/"):
            self._path_patterns.append(pattern)
        else:  # `secrets/**`: there is no folder to anchor it to, so it applies under any
            self._path_patterns.append(f"**/{pattern}")

    def _track(self) -> None:
        """Record the identity of every protected exact file and folder, and of the regular files under the
        folders (bounded: `MAX_TRACKED_FILES` files, `MAX_TRACKED_DIRS` folders, `MAX_TRACKED_DEPTH` levels)."""
        budget = _Budget()
        for root in self.home_roots():
            found = _identity(root)
            if found is None:
                continue
            file_id, mode = found
            if stat.S_ISREG(mode):
                self._file_ids.add(file_id)
            elif stat.S_ISDIR(mode):
                self._folder_ids.add(file_id)
                self._scan(root, lambda name: True, budget, frozenset(), MAX_TRACKED_DEPTH)

    def _track_named(self, workspace: Path) -> None:
        """Record the identity of the files in the workspace that the name patterns protect (`.env`, `*.pem`, ...),
        so a hard link made to one of them under another name is caught too. Same bounds, its own budget."""

        def wanted(name: str) -> bool:
            lowered = name.lower()
            if any(fnmatch.fnmatchcase(lowered, pattern) for pattern in self.extra_name_patterns):
                return True
            return lowered not in self.exempt_names and any(
                fnmatch.fnmatchcase(lowered, pattern) for pattern in self.builtin_name_patterns
            )

        self._scan(workspace, wanted, _Budget(), SCAN_SKIP, MAX_WORKSPACE_DEPTH)

    def _scan(
        self, root: Path, wanted: Callable[[str], bool], budget: _Budget, skip: frozenset[str], max_depth: int
    ) -> None:
        """Add the identity of the regular files under `root` whose name is `wanted`. Stops at the bounds and says so."""
        stack = [(root, 0)]
        while stack:
            folder, depth = stack.pop()
            if budget.dirs >= MAX_TRACKED_DIRS:
                self.tracking_truncated = True
                return
            budget.dirs += 1
            try:
                with os.scandir(folder) as entries:
                    listing = list(entries)
            except OSError:
                continue
            for entry in listing:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in skip:
                            continue
                        if depth >= max_depth:
                            self.tracking_truncated = True
                        else:
                            stack.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and wanted(entry.name):
                        if budget.files >= MAX_TRACKED_FILES:
                            self.tracking_truncated = True
                            return
                        info = entry.stat(follow_symlinks=False)
                        self._file_ids.add((info.st_dev, info.st_ino))
                        budget.files += 1
                except OSError:
                    continue

    @property
    def builtin_count(self) -> int:
        return len(BUILTIN_HOME) + len(BUILTIN_NAMES)

    @property
    def extra_count(self) -> int:
        return len(self._extra)

    def is_protected(self, path: Path, folders: dict[Path, bool] | None = None) -> bool:
        """`folders` is an optional per-call cache of folder lookups (a listing asks about many files in few folders);
        do not keep one across calls, a folder can be moved."""
        pure = PurePath(_canonical(str(path)))
        name = pure.name.lower()
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.extra_name_patterns):
            return True
        if name not in self.exempt_names and any(fnmatch.fnmatchcase(name, p) for p in self.builtin_name_patterns):
            return True
        if any(pure.full_match(pattern, case_sensitive=False) for pattern in self._path_patterns):
            return True
        return self._same_place(path, folders if folders is not None else {})

    def is_protected_fd(self, fd: int) -> bool:
        """Is the file open as `fd` a protected one? Checked on the open file itself, so a symlink swapped in after a
        path check cannot get past it: by `(st_dev, st_ino)` and by the real path the OS reports for the descriptor.
        Fails closed: a descriptor that cannot be examined counts as protected."""
        try:
            info = os.fstat(fd)
        except OSError:
            return True
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) in self._file_ids:
            return True
        real = fd_path(fd)
        return real is not None and self.is_protected(Path(real))

    def _same_place(self, path: Path, folders: dict[Path, bool]) -> bool:
        """By identity: `path` is a protected file or folder (or a hard link to one), or lies inside a folder."""
        if not self._file_ids and not self._folder_ids:
            return False
        found = _identity(path)
        if found is not None and found[0] in (self._file_ids if stat.S_ISREG(found[1]) else self._folder_ids):
            return True
        return self._inside_folder(path.parent, folders)

    def _inside_folder(self, folder: Path, folders: dict[Path, bool]) -> bool:
        if not self._folder_ids:
            return False
        known = folders.get(folder)
        if known is not None:
            return known
        found = _identity(folder)
        inside = found is not None and found[0] in self._folder_ids
        if not inside and folder.parent != folder:
            inside = self._inside_folder(folder.parent, folders)
        folders[folder] = inside
        return inside

    def patterns(self) -> list[str]:
        """Every pattern as written (built-in first), for `/guardrails`."""
        return [*BUILTIN_HOME, *BUILTIN_NAMES, *self._extra]

    def home_roots(self) -> list[Path]:
        """The absolute folders and files the path patterns name (a trailing `/**` removed).
        A pattern with other wildcards is left to the name rules."""
        roots: list[Path] = []
        for pattern in self._path_patterns:
            if not pattern.startswith("/"):  # `secrets/**` has no folder: it is matched by pattern, not by root
                continue
            base = pattern.removesuffix("/**")
            if base and not _WILDCARDS.intersection(base):
                root = Path(base)
                if root not in roots:
                    roots.append(root)
        return roots
