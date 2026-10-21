"""FastAPI application for BEV API, OpenAPI docs, and static UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from bev.config import static_dir as resolve_static_dir
from bev.server.routes import router
from bev.server.schemas import ErrorResponse
from bev.server.store import RunStore
from bev.config import resolve_accounts_db_path
from bev.store.account_store import AccountStore
from bev.store.workspace_store import bootstrap_workspace_store


def _error_response(status_code: int, error: str, hint: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=error, hint=hint)
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def create_app(
    scan_root: Path,
    store_dir: Path,
    *,
    accounts_db: Path | None = None,
) -> FastAPI:
    """Build the BEV API + static UI app.

    Args:
        scan_root: Path to W&B logs (see :func:`~bev.discovery.discover_wandb_runs`).
        store_dir: Central SQLite store for this BEV instance (all runs).
        accounts_db: Optional override for users / workspace prefs DB (tests).

    Returns:
        Configured FastAPI app (Swagger UI at ``/api/docs``).
    """
    app = FastAPI(
        title="BEV",
        description="Local W&B diagnostics — ingest, catalog, and chart APIs.",
        version="0.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.state.run_store = RunStore(scan_root, store_dir)
    account_store = AccountStore(accounts_db or resolve_accounts_db_path())
    app.state.account_store = account_store
    app.state.workspace_store = bootstrap_workspace_store(account_store, scan_root, store_dir)
    app.include_router(router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, str):
            return _error_response(exc.status_code, exc.detail)
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        _: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        if isinstance(exc.detail, str):
            return _error_response(exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(400, "invalid request")

    ui_static = resolve_static_dir()
    if ui_static.is_dir() and (ui_static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui_static, html=True), name="static")
    else:

        @app.get("/", include_in_schema=False)
        async def ui_missing() -> JSONResponse:
            return _error_response(
                503,
                "UI not built",
                hint="cd tutorials/pathologies/bev/webui && npm install && npm run build",
            )

    return app
