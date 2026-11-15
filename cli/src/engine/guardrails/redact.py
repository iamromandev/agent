"""Redaction of secrets in tool output. Phase 1 ships only the shape: it has no patterns yet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Redactor:
    env_values: Mapping[str, str]  # name -> value of the variables left out of child environments

    def redact(self, text: str) -> tuple[str, int]:
        """`(new text, number of replacements)`."""
        return text, 0
