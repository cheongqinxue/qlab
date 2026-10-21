"""Load the central ``agents/.env`` for local (non-Docker) runs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_AGENTS_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ENV_FILE = _AGENTS_ROOT / ".env"


def load_agents_env(*, env_file: Path | None = None) -> Path | None:
    """Load environment variables from the shared agents ``.env`` if it exists.

    Args:
        env_file: Override path; defaults to ``agents/.env``.

    Returns:
        Path that was loaded, or None if the file does not exist.
    """
    path = env_file or Path(os.environ.get("AGENTS_ENV_FILE", str(_DEFAULT_ENV_FILE)))
    if path.is_file():
        load_dotenv(path, override=False)
        return path
    return None
