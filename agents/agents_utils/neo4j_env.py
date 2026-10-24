"""Neo4j connection helpers (env-based, no hardcoded production secrets)."""

from __future__ import annotations

import os


def neo4j_password_from_env() -> str:
    """Return Neo4j password from environment.

    Returns:
        Password string.

    Raises:
        ValueError: If ``NEO4J_PASSWORD`` is unset or empty.
    """
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not password:
        msg = "NEO4J_PASSWORD must be set when using a Neo4j backend"
        raise ValueError(msg)
    return password
