"""Pydantic v2 models for the BEV HTTP API (OpenAPI source of truth)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Standard error body."""

    error: str
    hint: str | None = None


class RunSummary(BaseModel):
    """One run in ``GET /api/runs``."""

    wandb_run_id: str
    display_name: str | None = None
    project: str | None = None
    wandb_file: str
    store_version: int = 0
    last_step: int | None = None


class RunManifest(BaseModel):
    """``GET /api/runs/{run_id}/manifest``."""

    wandb_run_id: str | None = None
    display_name: str | None = None
    project: str | None = None
    store_version: int | None = None
    wandb_file: str | None = None
    checkpoint_dir: str | None = None
    last_step: int | None = None
    metric_count: int = 0


class MetricCatalogRow(BaseModel):
    """One row from ``metrics_catalog``."""

    key: str
    dtype: str | None = None
    first_step: int | None = None
    last_step: int | None = None
    n_points: int | None = None
    path_segments: list[str] | dict[str, Any] | None = None


class CatalogResponse(BaseModel):
    """``GET /api/runs/{run_id}/catalog`` (unfiltered key list only)."""

    metrics: list[MetricCatalogRow]
    match_count: int
    truncated: bool


class MetricSearchSummaryNode(BaseModel):
    """One node in a collapsed metric path summary tree."""

    label: str
    key_count: int
    children: list[MetricSearchSummaryNode] = Field(default_factory=list)
    ellipsis: bool = False
    collapsed: bool = False


class MetricSearchSummary(BaseModel):
    """Collapsed structure for all keys matching a search scope."""

    label: str
    key_count: int
    children: list[MetricSearchSummaryNode] = Field(default_factory=list)
    truncated: bool = False
    max_nodes: int = 500


class MetricsSearchResponse(BaseModel):
    """``GET /api/runs/{run_id}/metrics/search`` (regex / metric-scope discovery)."""

    metrics: list[MetricCatalogRow]
    match_count: int
    truncated: bool
    summary: MetricSearchSummary | None = None


class MetricFamily(BaseModel):
    """Scoped metric family summary (``section/metric``)."""

    id: str
    label: str
    plottable_count: int


class MetricFamiliesResponse(BaseModel):
    """``GET /api/runs/{run_id}/metric-families``."""

    families: list[MetricFamily]


class SeriesResponse(BaseModel):
    """``GET /api/runs/{run_id}/series``."""

    key: str
    x_axis: str
    points: list[list[float]]


class SeriesPrefixResponse(BaseModel):
    """``GET /api/runs/{run_id}/series/prefix``."""

    prefix: str
    series: dict[str, list[list[float]]]


class SeriesBatchRequest(BaseModel):
    """``POST /api/runs/{run_id}/series/batch`` body (known metric keys only)."""

    keys: list[str] = Field(..., min_length=1)
    max_points: int = Field(default=2000, ge=1)
    max_keys: int = Field(default=10_000, ge=1, le=10_000)
    x_axis: str = "step"
    scope: str = ""
    pool_by: str = ""
    pool_op: str = ""


class SeriesBatchResponse(BaseModel):
    """``POST /api/runs/{run_id}/series/batch``."""

    series: dict[str, list[list[float]]]
    match_count: int
    truncated: bool
    pattern: str = ""
    metric: str = ""
    path_pattern: str = ""
    scope: str = ""
    keys: list[str] = Field(default_factory=list)
    pool_op: str = ""
    pooled_series: list[list[float]] | None = None


class WorkspaceDocumentResponse(BaseModel):
    """``GET /api/workspace``."""

    sections: list[dict[str, Any]] = Field(default_factory=list)
    scan_root: str = ""
    username: str = ""


class WorkspaceDocumentRequest(BaseModel):
    """``PUT /api/workspace``."""

    sections: list[dict[str, Any]] = Field(default_factory=list)

