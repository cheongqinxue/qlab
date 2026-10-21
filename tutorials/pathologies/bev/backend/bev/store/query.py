"""Read paths for ingested run databases."""

from __future__ import annotations

import sqlite3
from typing import Any

_BUILTIN_X_AXES: frozenset[str] = frozenset({"step", "global_step", "epoch", "logged_at"})

_POOL_SQL_AGG: dict[str, str] = {"min": "MIN", "max": "MAX", "mean": "AVG"}


def run_manifest(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return polling metadata for a run store."""
    row = connection.execute(
        """
        SELECT wandb_run_id, display_name, project, store_version, wandb_file, checkpoint_dir
        FROM run_meta WHERE id = 1
        """
    ).fetchone()
    if row is None:
        return {}
    last_step = connection.execute(
        "SELECT MAX(step) FROM steps"
    ).fetchone()[0]
    metric_count = connection.execute(
        "SELECT COUNT(*) FROM metrics_catalog"
    ).fetchone()[0]
    return {
        "wandb_run_id": row[0],
        "display_name": row[1],
        "project": row[2],
        "store_version": row[3],
        "wandb_file": row[4],
        "checkpoint_dir": row[5],
        "last_step": last_step,
        "metric_count": int(metric_count or 0),
    }


def list_metric_keys(
    connection: sqlite3.Connection,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return catalog rows for the metric picker.

    Args:
        connection: Open run database.
        pattern: Optional Python regex matched against metric ``key`` strings.
        limit: Optional max rows (see :func:`~bev.store.catalog_query.query_metric_catalog`).
    """
    from bev.store.catalog_query import query_metric_catalog

    return query_metric_catalog(connection, pattern=pattern, limit=limit)["metrics"]


def fetch_scalar_at_step(
    connection: sqlite3.Connection,
    key: str,
    step: float,
) -> float | None:
    """Return the metric value at ``step``, or the nearest logged step.

    Args:
        connection: Open run database.
        key: Metric key.
        step: Training step to read.

    Returns:
        Scalar value or ``None`` when the key has no points.
    """
    rows = connection.execute(
        """
        SELECT mv.step,
               COALESCE(mv.value_real, CAST(mv.value_int AS REAL)) AS value
        FROM metric_values mv
        JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
        WHERE mc.key = ? AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
        ORDER BY mv.step
        """,
        (key,),
    ).fetchall()
    if not rows:
        return None
    best_value = None
    best_dist = float("inf")
    for logged_step, value in rows:
        if value is None:
            continue
        if logged_step == step:
            return float(value)
        dist = abs(float(logged_step) - step)
        if dist < best_dist:
            best_dist = dist
            best_value = float(value)
    return best_value


def fetch_scalars_at_step_bulk(
    connection: sqlite3.Connection,
    keys: list[str],
    step: float,
    *,
    chunk_size: int = 400,
) -> dict[str, float | None]:
    """Return nearest-at-step scalars for many metric keys.

    Args:
        connection: Open run database.
        keys: Metric keys to load.
        step: Training step to read.
        chunk_size: Max keys per SQL ``IN`` batch.

    Returns:
        Map of key to scalar or ``None`` when no points exist.
    """
    if not keys:
        return {}
    result: dict[str, float | None] = {key: None for key in keys}
    for offset in range(0, len(keys), chunk_size):
        chunk = keys[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT mc.key, mv.step,
                   COALESCE(mv.value_real, CAST(mv.value_int AS REAL)) AS value
            FROM metric_values mv
            JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
            WHERE mc.key IN ({placeholders})
              AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
            ORDER BY mc.key, mv.step
            """,
            chunk,
        ).fetchall()
        best_dist: dict[str, float] = {}
        for key, logged_step, value in rows:
            if value is None:
                continue
            if logged_step == step:
                result[key] = float(value)
                best_dist[key] = -1.0
                continue
            if best_dist.get(key) == -1.0:
                continue
            dist = abs(float(logged_step) - step)
            prev = best_dist.get(key)
            if prev is None or dist < prev:
                best_dist[key] = dist
                result[key] = float(value)
    return result


def normalize_pool_op(pool_op: str | None) -> str | None:
    """Return ``min``, ``max``, or ``mean`` when valid; otherwise ``None``."""
    if not pool_op:
        return None
    normalized = pool_op.strip().lower()
    return normalized if normalized in _POOL_SQL_AGG else None


def fetch_pooled_series_for_keys(
    connection: sqlite3.Connection,
    keys: list[str],
    pool_op: str,
    *,
    max_points: int = 2000,
    x_axis: str = "step",
) -> list[list[float]]:
    """Aggregate plottable values across keys at each x using one SQL ``GROUP BY``.

    Args:
        connection: Open run database.
        keys: Metric keys to combine (typically the batch key list).
        pool_op: ``min``, ``max``, or ``mean``.
        max_points: Downsample cap after aggregation.
        x_axis: Builtin step axis only (``metric:…`` x is not pooled yet).

    Returns:
        ``[[x, y], ...]`` downsampled series, or ``[]`` when pooling is unavailable.
    """
    op = normalize_pool_op(pool_op)
    if not op or not keys:
        return []
    if x_axis.startswith("metric:"):
        return []
    agg = _POOL_SQL_AGG[op]
    placeholders = ",".join("?" for _ in keys)
    params = list(keys)

    if x_axis == "step":
        rows = connection.execute(
            f"""
            SELECT mv.step AS x,
                   {agg}(COALESCE(mv.value_real, CAST(mv.value_int AS REAL))) AS y
            FROM metric_values mv
            JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
            WHERE mc.key IN ({placeholders})
              AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
            GROUP BY mv.step
            ORDER BY mv.step
            """,
            params,
        ).fetchall()
    elif x_axis in _BUILTIN_X_AXES:
        column = {
            "global_step": "s.global_step",
            "epoch": "s.epoch",
            "logged_at": "s.logged_at",
        }[x_axis]
        rows = connection.execute(
            f"""
            SELECT {column} AS x,
                   {agg}(COALESCE(mv.value_real, CAST(mv.value_int AS REAL))) AS y
            FROM metric_values mv
            JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
            JOIN steps s ON s.step = mv.step
            WHERE mc.key IN ({placeholders})
              AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
              AND {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY {column}
            """,
            params,
        ).fetchall()
    else:
        rows = connection.execute(
            f"""
            SELECT mv.step AS x,
                   {agg}(COALESCE(mv.value_real, CAST(mv.value_int AS REAL))) AS y
            FROM metric_values mv
            JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
            WHERE mc.key IN ({placeholders})
              AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
            GROUP BY mv.step
            ORDER BY mv.step
            """,
            params,
        ).fetchall()

    points = [(float(x_val), float(y_val)) for x_val, y_val in rows if y_val is not None]
    return _downsample(points, max_points)


def fetch_series(
    connection: sqlite3.Connection,
    key: str,
    *,
    max_points: int = 2000,
    x_axis: str = "step",
) -> list[list[float]]:
    """Return ``[[x, y], ...]`` downsampled for charting.

    Args:
        connection: Open run database.
        key: Metric key for the y-axis.
        max_points: Maximum points after downsampling.
        x_axis: ``step``, ``global_step``, ``epoch``, ``logged_at``, or
            ``metric:<other_key>`` to use another logged metric as x.
    """
    if x_axis.startswith("metric:"):
        x_key = x_axis.removeprefix("metric:")
        points = _fetch_series_with_metric_x(connection, y_key=key, x_key=x_key)
    elif x_axis in _BUILTIN_X_AXES:
        points = _fetch_series_builtin_x(connection, key=key, x_axis=x_axis)
    else:
        points = _fetch_series_builtin_x(connection, key=key, x_axis="step")
    return _downsample(points, max_points)


def _fetch_series_builtin_x(
    connection: sqlite3.Connection,
    *,
    key: str,
    x_axis: str,
) -> list[tuple[float, float]]:
    if x_axis == "step":
        rows = connection.execute(
            """
            SELECT mv.step,
                   COALESCE(mv.value_real, CAST(mv.value_int AS REAL)) AS value
            FROM metric_values mv
            JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
            WHERE mc.key = ? AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
            ORDER BY mv.step
            """,
            (key,),
        ).fetchall()
        return [(float(step), float(value)) for step, value in rows if value is not None]

    column = {
        "global_step": "s.global_step",
        "epoch": "s.epoch",
        "logged_at": "s.logged_at",
    }[x_axis]
    rows = connection.execute(
        f"""
        SELECT {column} AS x,
               COALESCE(mv.value_real, CAST(mv.value_int AS REAL)) AS value
        FROM metric_values mv
        JOIN metrics_catalog mc ON mc.metric_id = mv.metric_id
        JOIN steps s ON s.step = mv.step
        WHERE mc.key = ?
          AND (mv.value_real IS NOT NULL OR mv.value_int IS NOT NULL)
          AND {column} IS NOT NULL
        ORDER BY mv.step
        """,
        (key,),
    ).fetchall()
    return [(float(x_val), float(value)) for x_val, value in rows if value is not None]


def _fetch_series_with_metric_x(
    connection: sqlite3.Connection,
    *,
    y_key: str,
    x_key: str,
) -> list[tuple[float, float]]:
    rows = connection.execute(
        """
        SELECT COALESCE(x_mv.value_real, CAST(x_mv.value_int AS REAL)) AS x,
               COALESCE(y_mv.value_real, CAST(y_mv.value_int AS REAL)) AS y
        FROM metric_values y_mv
        JOIN metrics_catalog y_mc ON y_mc.metric_id = y_mv.metric_id
        JOIN metric_values x_mv
          ON x_mv.step = y_mv.step
        JOIN metrics_catalog x_mc ON x_mc.metric_id = x_mv.metric_id
        WHERE y_mc.key = ?
          AND x_mc.key = ?
          AND (y_mv.value_real IS NOT NULL OR y_mv.value_int IS NOT NULL)
          AND (x_mv.value_real IS NOT NULL OR x_mv.value_int IS NOT NULL)
        ORDER BY y_mv.step
        """,
        (y_key, x_key),
    ).fetchall()
    return [(float(x_val), float(y_val)) for x_val, y_val in rows]


def fetch_series_prefix(
    connection: sqlite3.Connection,
    prefix: str,
    *,
    max_points: int = 2000,
) -> dict[str, list[list[float]]]:
    """Return series for all keys starting with ``prefix``."""
    keys = [
        row[0]
        for row in connection.execute(
            "SELECT key FROM metrics_catalog WHERE key LIKE ? ORDER BY key",
            (f"{prefix}%",),
        ).fetchall()
    ]
    return {key: fetch_series(connection, key, max_points=max_points) for key in keys}


def _downsample(points: list[tuple[float, float]], max_points: int) -> list[list[float]]:
    if len(points) <= max_points:
        return [[step, value] for step, value in points]
    stride = max(1, len(points) // max_points)
    sampled = [points[index] for index in range(0, len(points), stride)]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return [[step, value] for step, value in sampled]
