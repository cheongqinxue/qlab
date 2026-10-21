"""Catalog filtering and batch reads for the HTTP API."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from bev.catalog.metric_scope import resolve_catalog_pattern
from bev.catalog.search_summary import build_search_summary_tree
from bev.store.family_ops import family_id_for_name
from bev.store.query import fetch_pooled_series_for_keys, fetch_series, normalize_pool_op


def _filter_keys_by_drill_scope(keys: list[str], scope: str) -> list[str]:
    """Keep keys that contain ``scope`` when non-empty."""
    trimmed = scope.strip()
    if not trimmed:
        return keys
    return [key for key in keys if trimmed in key]


def filter_catalog_rows_by_regex(
    rows: list[dict[str, Any]],
    regex: re.Pattern[str],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Keep rows whose ``key`` matches ``regex`` (``.search`` semantics).

    Uses one pass over ``rows``. When ``limit`` is set, only the first ``limit``
    matching rows are retained but ``match_count`` is the full number of matches.

    Line-joined ``findall`` is not used here: on CPython, many small ``.search``
    calls beat one multiline scan over a megabyte ``"\\n".join(keys)`` buffer.

    Args:
        rows: Catalog rows to filter.
        regex: Compiled user pattern.
        limit: Optional cap on returned rows (preview / apply).

    Returns:
        Filtered rows, total match count, and whether ``limit`` cut the list.
    """
    matched: list[dict[str, Any]] = []
    match_count = 0
    cap = None if limit is None or limit < 0 else limit
    for row in rows:
        if not regex.search(row["key"]):
            continue
        match_count += 1
        if cap is None or len(matched) < cap:
            matched.append(row)
    truncated = cap is not None and match_count > cap
    return matched, match_count, truncated


def compile_metric_pattern(pattern: str | None) -> re.Pattern[str] | None:
    """Compile a user regex or return ``None`` when empty / invalid."""
    if pattern is None:
        return None
    stripped = pattern.strip()
    if not stripped:
        return None
    try:
        return re.compile(stripped)
    except re.error:
        return None


def _catalog_rows(
    connection: sqlite3.Connection,
    *,
    family_id: int | None = None,
) -> list[dict[str, Any]]:
    if family_id is not None:
        rows = connection.execute(
            """
            SELECT key, dtype, first_step, last_step, n_points, path_segments
            FROM metrics_catalog
            WHERE family_id = ?
            ORDER BY key
            """,
            (family_id,),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT key, dtype, first_step, last_step, n_points, path_segments
            FROM metrics_catalog
            ORDER BY key
            """
        ).fetchall()
    return [
        {
            "key": key,
            "dtype": dtype,
            "first_step": first_step,
            "last_step": last_step,
            "n_points": n_points,
            "path_segments": json.loads(path_segments) if path_segments else None,
        }
        for key, dtype, first_step, last_step, n_points, path_segments in rows
    ]


def query_metric_catalog(
    connection: sqlite3.Connection,
    *,
    pattern: str | None = None,
    metric: str | None = None,
    path_pattern: str | None = None,
    limit: int | None = None,
    include_summary: bool = False,
) -> dict[str, Any]:
    """Filter metrics by regex with optional cap for previews.

    Args:
        connection: Open run database.
        pattern: Optional regex on metric keys (legacy full-key filter).
        metric: Scoped family ``section/metric`` (with optional ``path_pattern``).
        path_pattern: Regex on the path after ``metric/``.
        limit: Max rows returned (``match_count`` still reflects full match set).
        include_summary: When true, attach a collapsed path tree for all matches.

    Returns:
        ``metrics``, ``match_count``, ``truncated``, and optional ``summary``.
    """
    resolved = resolve_catalog_pattern(
        pattern=pattern,
        metric=metric,
        path_pattern=path_pattern,
    )
    if metric and metric.strip() and not (pattern and pattern.strip()):
        scoped_family_id = family_id_for_name(connection, metric)
        if scoped_family_id is None:
            return {
                "metrics": [],
                "match_count": 0,
                "truncated": False,
                "summary": None,
            }
        metrics = _catalog_rows(connection, family_id=scoped_family_id)
    else:
        metrics = _catalog_rows(connection)
    summary: dict[str, Any] | None = None
    if resolved is not None:
        regex = compile_metric_pattern(resolved)
        if regex is None:
            return {
                "metrics": [],
                "match_count": 0,
                "truncated": False,
                "summary": None,
            }
        if include_summary:
            matched, match_count, truncated = filter_catalog_rows_by_regex(
                metrics,
                regex,
                limit=None,
            )
            keys = [row["key"] for row in matched]
            summary = build_search_summary_tree(keys)
            preview_cap = 0 if limit == 0 else limit
            metrics = (
                matched[:preview_cap]
                if preview_cap is not None and preview_cap >= 0
                else []
            )
            if (
                preview_cap is not None
                and preview_cap >= 0
                and match_count > preview_cap
            ):
                truncated = True
            else:
                truncated = False
        else:
            metrics, match_count, truncated = filter_catalog_rows_by_regex(
                metrics,
                regex,
                limit=limit,
            )
    elif pattern is not None or metric is not None:
        return {"metrics": [], "match_count": 0, "truncated": False, "summary": None}
    else:
        match_count = len(metrics)
        truncated = False
        if limit is not None and limit >= 0 and match_count > limit:
            metrics = metrics[:limit]
            truncated = True
    return {
        "metrics": metrics,
        "match_count": match_count,
        "truncated": truncated,
        "summary": summary,
    }


def fetch_series_batch(
    connection: sqlite3.Connection,
    *,
    pattern: str | None = None,
    metric: str | None = None,
    path_pattern: str | None = None,
    max_points: int = 2000,
    x_axis: str = "step",
    max_keys: int = 512,
    scope: str = "",
) -> dict[str, Any]:
    """Return downsampled series for keys matching the resolved scope.

    Args:
        connection: Open run database.
        pattern: Legacy full-key regex.
        metric: Scoped family ``section/metric``.
        path_pattern: Path regex after ``metric/``.
        max_points: Per-series downsample cap.
        x_axis: Step axis id passed to :func:`fetch_series`.
        max_keys: Maximum number of series returned.

    Returns:
        ``series`` map, ``match_count``, and ``truncated`` flag.
    """
    if pattern is not None:
        raise DeprecationWarning(
            "`pattern` argument in fetch_series_batch has been deprecated."
        )
    catalog = query_metric_catalog(
        connection,
        pattern=pattern,
        metric=metric,
        path_pattern=path_pattern,
        limit=None,
    )
    keys = [
        row["key"] for row in catalog["metrics"] if row.get("dtype") in ("float", "int")
    ]
    keys = _filter_keys_by_drill_scope(keys, scope)
    match_count = len(keys)
    truncated = False
    if match_count > max_keys:
        keys = keys[:max_keys]
        truncated = True
    series: dict[str, list[list[float]]] = {}
    for key in keys:
        series[key] = fetch_series(
            connection, key, max_points=max_points, x_axis=x_axis
        )
    resolved = resolve_catalog_pattern(
        pattern=pattern,
        metric=metric,
        path_pattern=path_pattern,
    )
    return {
        "series": series,
        "match_count": match_count,
        "truncated": truncated,
        "pattern": resolved or "",
        "metric": (metric or "").strip(),
        "path_pattern": (path_pattern or "").strip(),
        "scope": scope.strip(),
        "keys": keys,
    }


def fetch_series_batch_for_keys(
    connection: sqlite3.Connection,
    keys: list[str],
    *,
    max_points: int = 2000,
    x_axis: str = "step",
    max_keys: int = 512,
    scope: str = "",
    pool_op: str = "",
    pool_by: str = "",
) -> dict[str, Any]:
    """Return series for an explicit key list (no regex catalog scan).

    Args:
        connection: Open run database.
        keys: Metric keys to load (order preserved; unknown keys skipped).
        max_points: Per-series downsample cap.
        x_axis: Step axis id passed to :func:`fetch_series`.
        max_keys: Maximum number of series returned.
        scope: Optional drill substring filter applied after key validation.
        pool_op: When ``min``, ``max``, or ``mean``, adds ``pooled_series`` via SQL
            ``GROUP BY`` x (``pool_by`` reserved; ignored until implemented).
        pool_by: Reserved facet for grouped pooling (ignored).

    Returns:
        Same shape as :func:`fetch_series_batch`, plus optional ``pooled_series``.
    """
    _ = pool_by
    normalized_pool = normalize_pool_op(pool_op)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        trimmed = key.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        deduped.append(trimmed)
    if not deduped:
        return {
            "series": {},
            "match_count": 0,
            "truncated": False,
            "pattern": "",
            "metric": "",
            "path_pattern": "",
            "scope": scope.strip(),
            "keys": [],
            "pool_op": "",
            "pooled_series": None,
        }

    placeholders = ",".join("?" for _ in deduped)
    rows = connection.execute(
        f"""
        SELECT key, dtype FROM metrics_catalog
        WHERE key IN ({placeholders})
        """,
        deduped,
    ).fetchall()
    allowed = {key: dtype for key, dtype in rows if dtype in ("float", "int")}
    ordered = [key for key in deduped if key in allowed]
    ordered = _filter_keys_by_drill_scope(ordered, scope)
    match_count = len(ordered)
    truncated = False
    if match_count > max_keys:
        ordered = ordered[:max_keys]
        truncated = True
    series: dict[str, list[list[float]]] = {}
    pooled_series: list[list[float]] | None = None
    if normalized_pool and ordered:
        pooled_series = fetch_pooled_series_for_keys(
            connection,
            ordered,
            normalized_pool,
            max_points=max_points,
            x_axis=x_axis,
        )
    if not normalized_pool:
        for key in ordered:
            series[key] = fetch_series(
                connection, key, max_points=max_points, x_axis=x_axis
            )
    return {
        "series": series,
        "match_count": match_count,
        "truncated": truncated,
        "pattern": "",
        "metric": "",
        "path_pattern": "",
        "scope": scope.strip(),
        "keys": ordered,
        "pool_op": normalized_pool or "",
        "pooled_series": pooled_series,
    }
