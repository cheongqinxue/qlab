"""``metrics_family`` helpers (populated during ingest)."""

from __future__ import annotations

import sqlite3

from bev.catalog.metric_scope import metric_family_from_key


def is_plottable_dtype(dtype: str) -> bool:
    """Return whether catalog dtype counts toward ``plottable_count``."""
    return dtype in ("float", "int")


def ensure_family_for_key(
    connection: sqlite3.Connection,
    key: str,
    *,
    dtype: str,
    count_plottable: bool,
) -> int | None:
    """Ensure a ``metrics_family`` row exists for scoped ``section/metric/…`` keys.

    Args:
        connection: Open SQLite connection.
        key: Full metric key.
        dtype: Catalog dtype for the metric.
        count_plottable: When True, increment ``plottable_count`` (new plottable key).

    Returns:
        ``family_id`` or ``None`` when the key has no scoped family.
    """
    family_name = metric_family_from_key(key)
    if family_name is None:
        return None
    connection.execute(
        "INSERT OR IGNORE INTO metrics_family (name, plottable_count) VALUES (?, 0)",
        (family_name,),
    )
    row = connection.execute(
        "SELECT family_id FROM metrics_family WHERE name = ?",
        (family_name,),
    ).fetchone()
    if row is None:
        return None
    family_id = int(row[0])
    if count_plottable and is_plottable_dtype(dtype):
        connection.execute(
            """
            UPDATE metrics_family
            SET plottable_count = plottable_count + 1
            WHERE family_id = ?
            """,
            (family_id,),
        )
    return family_id


def family_id_for_name(connection: sqlite3.Connection, name: str) -> int | None:
    """Look up ``family_id`` by metric family segment name."""
    row = connection.execute(
        "SELECT family_id FROM metrics_family WHERE name = ?",
        (name.strip(),),
    ).fetchone()
    return int(row[0]) if row else None


def list_metric_families(connection: sqlite3.Connection) -> list[dict[str, object]]:
    """Return sidebar-ready family rows from ``metrics_family``."""
    rows = connection.execute(
        """
        SELECT name, plottable_count
        FROM metrics_family
        WHERE plottable_count > 0
        ORDER BY name
        """
    ).fetchall()
    return [
        {"id": name, "label": name, "plottable_count": int(count)}
        for name, count in rows
    ]
