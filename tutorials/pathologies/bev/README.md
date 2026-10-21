# BEV (prototype)

Local W&B run viewer: ingest `run-*.wandb` into SQLite, explore metrics in the browser.

## Layout

| Path | Role |
|------|------|
| `backend/bev/` | Python package (`python -m bev`) — ingest, store, API |
| `backend/tests/` | Pytest for backend |
| `webui/` | Svelte 5 + Vite app |
| `e2e/` | Playwright specs + serve harness |
| `sdk/bev_sdk/` | Companion SDK (training hooks; placeholder) |
| `static/` | Production UI build (`webui` → `../static`) |
| `docs/` | API / design notes |

## Build the UI (once)

```bash
cd tutorials/pathologies/bev/webui
npm install
npm run build
```

Output is written to `tutorials/pathologies/bev/static/` (served by the Python app).

## Run

**Shortcut (pathology `.data` runs):**

```bash
cd /path/to/qlab
./tutorials/pathologies/bev/run.sh
./tutorials/pathologies/bev/run.sh encoder_mlm_explode
```

Defaults: `encoder_mlm_repeat_shift_adam_collapse`, store `~/.bev/store`, `--watch`, `--open`.

**Full control:**

```bash
./tutorials/pathologies/bev/scripts/serve.sh \
  --root tutorials/pathologies/.data/<RUN_ID>/wandb \
  --store-dir ~/.bev/store \
  --watch --open
```

(`serve.sh` sets `PYTHONPATH` to `backend/` so `python -m bev` resolves in-repo.)

- UI: http://127.0.0.1:8765/
- **Store (central, per BEV instance):** `~/.bev/store/runs/<wandb_run_id>/run.db` by default
- Override: `--store-dir /path/to/store` or `BEV_STORE_DIR`
- **Accounts / workspace prefs:** `~/.bev/accounts.db` (or `BEV_ACCOUNTS_DB`) — users, scan roots,
  panel layout per `(user, scan_root)`. No login UI yet; loads bootstrap **admin** (default password
  `password`, override with `BEV_ADMIN_PASSWORD`).
- `--root` is the **workspace identity** (W&B scan path) and controls **where to scan** for `.wandb` files
- `--watch` re-ingests when the `.wandb` file grows during training

## Dev mode (hot reload UI)

Terminal 1:

```bash
export PYTHONPATH=tutorials/pathologies/bev/backend
uv run python -m bev serve --root <wandb-dir> --watch --port 8765
```

Terminal 2:

```bash
cd tutorials/pathologies/bev/webui && npm run dev
```

Open http://localhost:5173 (API proxied to 8765).

## Backend reference

HTTP routes and per-run SQLite schema (FastAPI + Pydantic in `backend/bev/server/`):

- [`docs/backend-reference.md`](docs/backend-reference.md)
- Swagger UI: `http://127.0.0.1:8765/api/docs` (when `bev serve` is running)
- OpenAPI: `docs/openapi.json` (regenerate via `scripts/export_openapi.py`)

## Fixtures

Committed offline W&B scan roots live under `backend/bev/fixtures/data/`.
Regenerate: `python -m bev.fixtures <name>` (with `PYTHONPATH=backend/` or use `scripts/ensure-fixtures.sh`).

Playwright uses `e2e_minimal` by default (`BEV_SCAN_ROOT` to override).

## Tests

**Python (API, ingest, layout logic):**

```bash
uv run pytest tutorials/pathologies/bev/backend/tests tutorials/pathologies/bev/sdk/tests
```

**UI (Playwright, headless Chromium):**

```bash
cd tutorials/pathologies/bev/e2e
npm install
npx playwright install chromium
npm test
```

Playwright runs `e2e/serve.sh` against `backend/bev/fixtures/data/e2e_minimal`. To run everything:

```bash
./tutorials/pathologies/bev/scripts/test-all.sh
```
