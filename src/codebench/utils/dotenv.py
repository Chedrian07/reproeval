"""Minimal .env file loader — no external dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Parse a .env file and inject its values into ``os.environ``.

    Existing environment variables are NOT overwritten.
    Returns the dict of values that were actually set.
    """
    if path is None:
        path = Path.cwd() / ".env"
    if not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
