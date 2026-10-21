"""``metrics_catalog`` and ``metric_values`` writers."""

from __future__ import annotations

import sqlite3
from typing import Any

from bev.catalog.path_analyzer import analyze_path, infer_dtype
from bev.store.family_ops import ensure_family_for_key


def ensure_metric_in_catalog(
    connection: sqlite3.Connection,
    *,
    key: str,
    step: int,
    value: Any,
) -> int:
    """Insert or update catalog metadata and return ``metric_id``.

    Args:
        connection: Open SQLite connection.
        key: Full metric key from W&B history.
        step: W&B ``_step`` for this history row.
        value: Parsed metric value.

    Returns:
        ``metric_id`` for ``key``.
    """
    dtype = infer_dtype(value)
    row = connection.execute(
        "SELECT metric_id FROM metrics_catalog WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        family_id = ensure_family_for_key(
            connection,
            key,
            dtype=dtype,
            count_plottable=True,
        )
        connection.execute(
            """
            INSERT INTO metrics_catalog (
              key, dtype, first_step, last_step, n_points, path_segments, family_id
            ) VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (key, dtype, step, step, analyze_path(key), family_id),
        )
        metric_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        return metric_id
    metric_id = int(row[0])
    existing = connection.execute(
        "SELECT family_id FROM metrics_catalog WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()
    family_id = int(existing[0]) if existing and existing[0] is not None else None
    if family_id is None:
        family_id = ensure_family_for_key(
            connection,
            key,
            dtype=dtype,
            count_plottable=False,
        )
        if family_id is not None:
            connection.execute(
                "UPDATE metrics_catalog SET family_id = ? WHERE metric_id = ?",
                (family_id, metric_id),
            )
            if dtype in ("float", "int"):
                connection.execute(
                    """
                    UPDATE metrics_family
                    SET plottable_count = plottable_count + 1
                    WHERE family_id = ?
                    """,
                    (family_id,),
                )
    connection.execute(
        """
        UPDATE metrics_catalog
        SET last_step = ?, dtype = COALESCE(dtype, ?)
        WHERE metric_id = ?
        """,
        (step, dtype, metric_id),
    )
    return metric_id


def upsert_metric_value(
    connection: sqlite3.Connection,
    *,
    metric_id: int,
    step: int,
    value: Any,
) -> bool:
    """Write a scalar value; bump ``n_points`` only on first insert for this step.

    Args:
        connection: Open SQLite connection.
        metric_id: Catalog id.
        step: W&B step.
        value: Parsed JSON value.

    Returns:
        True if this was a new ``(metric_id, step)`` row.
    """
    value_real: float | None = None
    value_int: int | None = None
    value_text: str | None = None
    if isinstance(value, bool):
        value_int = int(value)
    elif isinstance(value, int):
        value_int = value
    elif isinstance(value, float):
        value_real = value
    elif isinstance(value, str):
        value_text = value
    elif value is None:
        pass
    else:
        value_text = str(value)

    existing = connection.execute(
        "SELECT 1 FROM metric_values WHERE metric_id = ? AND step = ?",
        (metric_id, step),
    ).fetchone()
    connection.execute(
        """
        INSERT INTO metric_values (metric_id, step, value_real, value_int, value_text)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(metric_id, step) DO UPDATE SET
          value_real = excluded.value_real,
          value_int = excluded.value_int,
          value_text = excluded.value_text
        """,
        (metric_id, step, value_real, value_int, value_text),
    )
    if existing is None:
        connection.execute(
            """
            UPDATE metrics_catalog
            SET n_points = n_points + 1,
                first_step = CASE
                  WHEN first_step IS NULL THEN ?
                  WHEN ? < first_step THEN ?
                  ELSE first_step
                END,
                last_step = CASE
                  WHEN last_step IS NULL THEN ?
                  WHEN ? > last_step THEN ?
                  ELSE last_step
                END
            WHERE metric_id = ?
            """,
            (step, step, step, step, step, step, metric_id),
        )
        return True
    connection.execute(
        """
        UPDATE metrics_catalog
        SET last_step = MAX(COALESCE(last_step, ?), ?)
        WHERE metric_id = ?
        """,
        (step, step, metric_id),
    )
    return False


def list_series(
    connection: sqlite3.Connection,
    key: str,
) -> list[tuple[int, float | None]]:
    """Return ``(step, value)`` pairs for a metric key.

    Args:
        connection: Open SQLite connection.
        key: Metric key in ``metrics_catalog``.

    Returns:
        Points ordered by ``step``.
    """
    rows = connection.execute(
        """
        SELECT mv.step, mv.value_real, mv.value_int
        FROM metric_values mv
        JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
        WHERE mc.key = ?
        ORDER BY mv.step
        """,
        (key,),
    ).fetchall()
    points: list[tuple[int, float | None]] = []
    for step, value_real, value_int in rows:
        if value_real is not None:
            points.append((int(step), float(value_real)))
        elif value_int is not None:
            points.append((int(step), float(value_int)))
        else:
            points.append((int(step), None))
    return points
