"""BEV command-line interface."""

from __future__ import annotations

import argparse
import logging
import webbrowser
from pathlib import Path

import uvicorn

from bev.config import resolve_store_dir
from bev.server.app import create_app
from bev.server.watcher import ingest_all, start_wandb_watcher


def _serve(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    store_dir = resolve_store_dir(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s] %(message)s")
    ingest_all(root, store_dir)
    observer = start_wandb_watcher(root, store_dir) if args.watch else None
    app = create_app(root, store_dir)
    url = f"http://{args.host}:{args.port}/"
    if args.open:
        webbrowser.open(url)
    print(f"BEV UI: {url}")
    print(f"Scan root: {root}")
    print(f"Store dir: {store_dir}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if observer is not None:
            observer.stop()
            observer.join()


def main() -> None:
    """Entry point for ``python -m bev``."""
    parser = argparse.ArgumentParser(prog="bev", description="Local W&B diagnostics UI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Ingest local runs and start the web UI")
    serve.add_argument(
        "--root",
        required=True,
        help="W&B directory to scan (any layout; see discover_wandb_runs)",
    )
    serve.add_argument(
        "--store-dir",
        default=None,
        help="Central SQLite store (default: BEV_STORE_DIR or ~/.bev/store)",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--watch", action="store_true", help="Re-ingest when .wandb grows")
    serve.add_argument("--open", action="store_true", help="Open browser")
    serve.set_defaults(func=_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
