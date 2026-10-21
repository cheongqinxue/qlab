"""FastAPI routes for the BEV HTTP API."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from bev.catalog.metric_scope import list_metric_families, resolve_catalog_pattern
from bev.discovery import discover_wandb_runs, run_db_path
from bev.ingest.pipeline import ingest_wandb_file
from bev.server.schemas import (
    CatalogResponse,
    MetricsSearchResponse,
    MetricFamiliesResponse,
    MetricFamily,
    RunManifest,
    RunSummary,
    SeriesBatchRequest,
    SeriesBatchResponse,
    SeriesPrefixResponse,
    SeriesResponse,
    WorkspaceDocumentRequest,
    WorkspaceDocumentResponse,
)
from bev.server.store import RunStore
from bev.store.catalog_query import fetch_series_batch_for_keys, query_metric_catalog
from bev.store.workspace_store import WorkspaceStore
from bev.store.query import fetch_series, fetch_series_prefix, run_manifest


def get_run_store(request: Request) -> RunStore:
    """Dependency: BEV store bound at app creation."""
    return request.app.state.run_store


def get_workspace_store(request: Request) -> WorkspaceStore:
    """Dependency: workspace document store for this BEV instance."""
    return request.app.state.workspace_store


RunStoreDep = Annotated[RunStore, Depends(get_run_store)]
WorkspaceStoreDep = Annotated[WorkspaceStore, Depends(get_workspace_store)]

router = APIRouter(prefix="/api", tags=["runs"])


def _run_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="run not found")


def _scope_required() -> HTTPException:
    return HTTPException(status_code=400, detail="metric or pattern required")


@router.get("/runs", response_model=list[RunSummary])
def list_runs(store: RunStoreDep) -> list[RunSummary]:
    """List runs under the scan root; ingest any run missing ``run.db``."""
    payload: list[RunSummary] = []
    for run in discover_wandb_runs(store.scan_root):
        db_path = run_db_path(store.store_dir, run.wandb_run_id)
        if not db_path.is_file():
            ingest_wandb_file(run.wandb_file, db_path).close()
        manifest: dict[str, Any] = {}
        if connection := store.connection(run.wandb_run_id):
            manifest = run_manifest(connection)
            connection.close()
        payload.append(
            RunSummary(
                wandb_run_id=run.wandb_run_id,
                display_name=manifest.get("display_name") or run.display_name,
                project=manifest.get("project"),
                wandb_file=str(run.wandb_file),
                store_version=int(manifest.get("store_version") or 0),
                last_step=manifest.get("last_step"),
            )
        )
    return payload


@router.get("/runs/{run_id}/manifest", response_model=RunManifest)
def get_run_manifest(run_id: str, store: RunStoreDep) -> RunManifest:
    """Polling metadata including ``metric_count`` and ``store_version``."""
    try:
        connection = store.connection_or_404(run_id)
    except LookupError:
        raise _run_not_found() from None
    try:
        return RunManifest.model_validate(run_manifest(connection))
    finally:
        connection.close()


@router.get("/runs/{run_id}/catalog", response_model=CatalogResponse)
def get_run_catalog(
    run_id: str,
    store: RunStoreDep,
    pattern: str | None = None,
    metric: str | None = None,
    path_pattern: str | None = None,
    limit: int | None = None,
) -> CatalogResponse:
    """Full metric catalog for small runs (no regex filters)."""
    if (pattern and pattern.strip()) or (metric and metric.strip()) or (
        path_pattern and path_pattern.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="use GET /api/runs/{run_id}/metrics/search for filtered metric discovery",
        )
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        return CatalogResponse.model_validate(
            query_metric_catalog(connection, limit=limit)
        )
    finally:
        connection.close()


@router.get("/runs/{run_id}/metrics/search", response_model=MetricsSearchResponse)
def search_run_metrics(
    run_id: str,
    store: RunStoreDep,
    pattern: str | None = None,
    metric: str | None = None,
    path_pattern: str | None = None,
    limit: int | None = None,
    summary: int = Query(default=0, ge=0, le=1),
) -> MetricsSearchResponse:
    """Discover metric keys by regex or scoped family filters (no series data)."""
    if not resolve_catalog_pattern(
        pattern=pattern,
        metric=metric,
        path_pattern=path_pattern,
    ):
        raise _scope_required()
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        include_summary = bool(summary)
        effective_limit = 0 if include_summary else limit
        return MetricsSearchResponse.model_validate(
            query_metric_catalog(
                connection,
                pattern=pattern,
                metric=metric,
                path_pattern=path_pattern,
                limit=effective_limit,
                include_summary=include_summary,
            )
        )
    finally:
        connection.close()


@router.get("/runs/{run_id}/metric-families", response_model=MetricFamiliesResponse)
def get_metric_families(run_id: str, store: RunStoreDep) -> MetricFamiliesResponse:
    """Distinct scoped ``section/metric`` families for plottable keys."""
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        families = [MetricFamily.model_validate(row) for row in list_metric_families(connection)]
        return MetricFamiliesResponse(families=families)
    finally:
        connection.close()


@router.get("/runs/{run_id}/series", response_model=SeriesResponse)
def get_series(
    run_id: str,
    store: RunStoreDep,
    key: str = "",
    max_points: int = Query(default=2000, ge=1),
    x_axis: str = "step",
) -> SeriesResponse:
    """Single downsampled time series."""
    if not key:
        raise HTTPException(status_code=400, detail="key required")
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        points = fetch_series(connection, key, max_points=max_points, x_axis=x_axis)
        return SeriesResponse(key=key, x_axis=x_axis, points=points)
    finally:
        connection.close()


@router.get("/runs/{run_id}/series/prefix", response_model=SeriesPrefixResponse)
def get_series_prefix(
    run_id: str,
    store: RunStoreDep,
    prefix: str = "",
    max_points: int = Query(default=2000, ge=1),
) -> SeriesPrefixResponse:
    """All series whose keys start with ``prefix``."""
    if not prefix:
        raise HTTPException(status_code=400, detail="prefix required")
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        series = fetch_series_prefix(connection, prefix, max_points=max_points)
        return SeriesPrefixResponse(prefix=prefix, series=series)
    finally:
        connection.close()


@router.post("/runs/{run_id}/series/batch", response_model=SeriesBatchResponse)
def post_series_batch(
    run_id: str,
    body: SeriesBatchRequest,
    store: RunStoreDep,
) -> SeriesBatchResponse:
    """Batch time series for an explicit list of known metric keys."""
    connection = store.connection(run_id)
    if connection is None:
        raise _run_not_found()
    try:
        pool_op = (body.pool_op or "").strip().lower()
        if pool_op and pool_op not in ("min", "max", "mean"):
            raise HTTPException(
                status_code=400,
                detail="pool_op must be min, max, mean, or empty",
            )
        payload = fetch_series_batch_for_keys(
            connection,
            body.keys,
            max_points=body.max_points,
            x_axis=body.x_axis,
            max_keys=body.max_keys,
            scope=body.scope.strip(),
            pool_op=pool_op,
            pool_by=(body.pool_by or "").strip(),
        )
        return SeriesBatchResponse.model_validate(payload)
    finally:
        connection.close()


@router.get("/workspace", response_model=WorkspaceDocumentResponse)
def get_workspace(workspace: WorkspaceStoreDep) -> WorkspaceDocumentResponse:
    """Load persisted workspace sections and panels."""
    return WorkspaceDocumentResponse(
        sections=workspace.load_sections(),
        scan_root=str(workspace.scan_root),
        username=workspace.username,
    )


@router.put("/workspace", response_model=WorkspaceDocumentResponse)
def put_workspace(
    body: WorkspaceDocumentRequest,
    workspace: WorkspaceStoreDep,
) -> WorkspaceDocumentResponse:
    """Save workspace sections and panels."""
    workspace.save_sections(body.sections)
    return WorkspaceDocumentResponse(
        sections=body.sections,
        scan_root=str(workspace.scan_root),
        username=workspace.username,
    )

