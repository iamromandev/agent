"""The environment handed to child processes: secret-looking variables are left out (R8)."""

from __future__ import annotations

import re
from collections.abc import Mapping

SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|CREDENTIAL)|(_KEY|_KEY_ID)$|^AWS_", re.IGNORECASE)
AWS_KEEP = frozenset({"AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"})
MIN_SECRET_LENGTH = 8  # a shorter value is too likely to be an ordinary word to redact it from output


class EnvFilter:
    def __init__(self, keep: frozenset[str]) -> None:
        """`keep`: upper-cased names that stay (global `keepEnv`); one ending in `*` is a prefix (`AWS_*`)."""
        self._keep = frozenset(name for name in keep if not name.endswith("*"))
        self._keep_prefixes = tuple(name[:-1] for name in keep if name.endswith("*"))

    def _kept(self, upper: str) -> bool:
        return upper in self._keep or upper in AWS_KEEP or upper.startswith(self._keep_prefixes)

    def _removes(self, name: str) -> bool:
        return SECRET_NAME.search(name) is not None and not self._kept(name.upper())

    def filter(self, env: Mapping[str, str]) -> dict[str, str]:
        return {name: value for name, value in env.items() if not self._removes(name)}

    def removed(self, env: Mapping[str, str]) -> list[str]:
        """The names `filter` leaves out, sorted. Never a value."""
        return sorted(name for name in env if self._removes(name))

    def secret_values(self, env: Mapping[str, str]) -> dict[str, str]:
        """`{name: value}` of the removed variables whose value is long enough to look for in output."""
        return {name: env[name] for name in self.removed(env) if len(env[name]) >= MIN_SECRET_LENGTH}
