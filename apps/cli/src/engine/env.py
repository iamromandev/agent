"""The environment of the processes the tools start."""

from __future__ import annotations

import os


def child_environment() -> dict[str, str]:
    """The parent's environment, asking tools for color (the screen shows it; the model gets it stripped)."""
    env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
    return env | {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"}
