# BEV backend reference

Authoritative implementation: `backend/bev/server/routes.py` + Pydantic models in
`backend/bev/server/schemas.py`, SQLite DDL in `backend/bev/store/schema.py`.

**Interactive docs (Swagger):** `http://127.0.0.1:8765/api/docs` (with `bev serve`)

**OpenAPI JSON:** `GET /api/openapi.json` — regenerate committed copy:

```bash
PYTHONPATH=tutorials/pathologies/bev/backend \
  uv run python tutorials/pathologies/bev/scripts/export_openapi.py
```

**Base URL:** `http://127.0.0.1:8765` (default `bev serve`)

**JSON:** `Content-Type: application/json` on all `/api/*` responses unless noted.

Human-readable field tables below mirror the Pydantic models; when in doubt, prefer
the live OpenAPI schema.

---

## Conventions

| Item | Rule |
|------|------|
| `{run_id}` | W&B run id (`wandb_run_id` from discovery / ingest) |
| Metric scope | Either legacy **`pattern`** (regex on full key) **or** **`metric`** + optional **`path_pattern`** (regex on path after scoped `section/metric/`) |
| Polling | Compare `store_version` from manifest; refetch catalog / series when it changes (bumped only when ingest applies new history events) |
| Ingest | `GET /api/runs` and missing `run.db` trigger ingest from the scan root |
| Workspace | Layout keyed by **`scan_root`** + user in **`~/.bev/accounts.db`** via **`GET/PUT /api/workspace`** (no login UI yet; session is bootstrap **admin**) |
| Accounts DB | **`BEV_ACCOUNTS_DB`** or default **`~/.bev/accounts.db`** — users, scan-root history, per-user workspace JSON |

---

## Error responses

Most failures return a JSON object:

```json
{ "error": "<message>" }
```

| HTTP | When | Example `error` |
|------|------|-----------------|
| `400` | Bad query/body | `key required`, `metric or pattern required`, use metrics search for filtered catalog |
| `422` | Validation | Empty `keys` on series batch |
| `404` | Unknown run or no DB | `run not found` |
| `503` | UI not built | `UI not built` (+ `hint` string) |

---

## Schemas

### `RunSummary`

Used in **`GET /api/runs`** response (top-level **JSON array** of these objects).

| Field | Type | Description |
|-------|------|-------------|
| `wandb_run_id` | string | Run id |
| `display_name` | string \| null | Display name |
| `project` | string \| null | W&B project |
| `wandb_file` | string | Absolute path to `run-*.wandb` |
| `store_version` | integer | Bumped on ingest (cache key) |
| `last_step` | integer \| null | Max logged step |

**Example (one element):**

```json
{
  "wandb_run_id": "beve2e1",
  "display_name": "e2e-minimal",
  "project": "bev_e2e",
  "wandb_file": "/path/to/run-xxx.wandb",
  "store_version": 3,
  "last_step": 20
}
```

### `RunManifest`

**`GET /api/runs/{run_id}/manifest`** → `200` body.

| Field | Type | Description |
|-------|------|-------------|
| `wandb_run_id` | string | |
| `display_name` | string \| null | |
| `project` | string \| null | |
| `store_version` | integer | |
| `wandb_file` | string \| null | |
| `checkpoint_dir` | string \| null | Heuristic link to checkpoints |
| `last_step` | integer \| null | |
| `metric_count` | integer | Rows in `metrics_catalog` |

```json
{
  "wandb_run_id": "beve2e1",
  "display_name": "e2e-minimal",
  "project": "bev_e2e",
  "store_version": 3,
  "wandb_file": "/path/to/run-xxx.wandb",
  "checkpoint_dir": null,
  "last_step": 20,
  "metric_count": 42
}
```

### `MetricCatalogRow`

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Full W&B metric key |
| `dtype` | string | `float`, `int`, `string`, `null`, … |
| `first_step` | integer \| null | |
| `last_step` | integer \| null | |
| `n_points` | integer | |
| `path_segments` | array of string \| null | Parsed path segments (metadata) |

### `CatalogResponse`

**`GET /api/runs/{run_id}/catalog`** → `200` body (unfiltered list; no regex query params).

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `MetricCatalogRow[]` | Rows returned (may be capped by `limit`) |
| `match_count` | integer | Total keys in catalog (before `limit`) |
| `truncated` | boolean | `true` if `limit` cut the `metrics` array |

### `MetricsSearchResponse`

**`GET /api/runs/{run_id}/metrics/search`** → `200` body (regex / metric-scope discovery only).

Same fields as `CatalogResponse`, plus optional **`summary`** (collapsed path tree). Requires
scope: non-empty `pattern` **or** non-empty `metric`.

```json
{
  "metrics": [
    {
      "key": "train/loss",
      "dtype": "float",
      "first_step": 0,
      "last_step": 20,
      "n_points": 21,
      "path_segments": ["train", "loss"]
    }
  ],
  "match_count": 1,
  "truncated": false
}
```

### `MetricFamily`

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Family name (segment after `train/`) |
| `label` | string | Same as `id` today |
| `plottable_count` | integer | `float`/`int` keys in that family |

### `MetricFamiliesResponse`

**`GET /api/runs/{run_id}/metric-families`** → `200` body.

```json
{
  "families": [
    { "id": "loss", "label": "loss", "plottable_count": 1 },
    { "id": "grad_norm", "label": "grad_norm", "plottable_count": 2 }
  ]
}
```

### `SeriesResponse`

**`GET /api/runs/{run_id}/series`** → `200` body.

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Echo of query param |
| `x_axis` | string | `step`, `global_step`, `epoch`, or `logged_at` |
| `points` | array of `[number, number]` | `[x, y]` pairs, downsampled |

```json
{
  "key": "train/loss",
  "x_axis": "step",
  "points": [[0, 1.2], [1, 1.1], [2, 1.05]]
}
```

### `SeriesPrefixResponse`

**`GET /api/runs/{run_id}/series/prefix`** → `200` body.

| Field | Type | Description |
|-------|------|-------------|
| `prefix` | string | Echo of query param |
| `series` | object | Map of metric key → `points` (same shape as `SeriesResponse.points`) |

```json
{
  "prefix": "train/grad_norm/",
  "series": {
    "train/grad_norm/layer.0": [[0, 0.5], [1, 0.25]]
  }
}
```

### `SeriesBatchRequest`

**`POST /api/runs/{run_id}/series/batch`** body (known metric keys only).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `keys` | string[] | yes | — | Metric keys to load |
| `max_keys` | integer | no | `10000` | Max series returned (hard cap `10000`) |
| `max_points` | integer | no | `2000` | Per-series downsample cap |
| `x_axis` | string | no | `step` | X axis for all series |
| `scope` | string | no | `""` | Substring filter on `keys` |
| `row_facet` | string | no | — | With `col_facet`, enables `facet_labels` |
| `col_facet` | string | no | — | |

### `SeriesBatchResponse`

**`POST /api/runs/{run_id}/series/batch`** → `200` body.

| Field | Type | Description |
|-------|------|-------------|
| `series` | object | Key → `points` (`[x,y][]`) |
| `match_count` | integer | Keys loaded (after `scope` filter and `max_keys` cap) |
| `keys` | string[] | Keys actually loaded (subset of request) |
| `truncated` | boolean | `true` if `match_count > max_keys` |
| `pattern` | string | Resolved full-key regex (may be synthesized from metric scope) |
| `metric` | string | Echo |
| `path_pattern` | string | Echo |
| `scope` | string | Echo |
| `row_facet` | string | Echo |
| `col_facet` | string | Echo |
| `facet_labels` | object | Key → `{ "row": string, "col": string }` (empty if facets not set) |

```json
{
  "series": {
    "train/loss": [[0, 1.0], [1, 0.9]]
  },
  "match_count": 1,
  "truncated": false,
  "pattern": "train/loss",
  "metric": "",
  "path_pattern": "",
  "scope": "",
  "row_facet": "",
  "col_facet": "",
  "facet_labels": {}
}
```

### `WorkspaceDocumentResponse` / `WorkspaceDocumentRequest`

**`GET /api/workspace`** and **`PUT /api/workspace`** share the same JSON shape.

| Field | Type | Description |
|-------|------|-------------|
| `sections` | array | Workspace sections; each section has `id`, `name`, `isOpen`, `columns`, `panels` |
| `scan_root` | string | Absolute W&B scan root for this server (`bev serve --root`); workspace identity |
| `username` | string | Active user (always **`admin`** until sign-in exists) |

Panel objects are opaque to the API (stored as JSON). The web UI includes viz type, titles,
metric keys, regex / metric-scope fields, heatmap options, Y-axis settings, and multipanel
**committed selection** (`comparisonSelectionKeys`, `comparisonSelectionRunId`,
`comparisonSelectionStoreVersion`, `comparisonSelectionFingerprint`,
`comparisonSelectionCommitted`). See `webui/src/lib/workspace.js` for the canonical field list.

```json
{
  "sections": [
    {
      "id": "…",
      "name": "Metrics",
      "isOpen": true,
      "columns": 2,
      "panels": []
    }
  ]
}
```

Each **`(user_id, scan_root)`** pair has its own `sections_json`. The UI does not offer
workspace switching yet: one process is bound to one `--root`, and the API loads the bootstrap
**admin** user automatically (password stored hashed; default plain text **`password`** via
``BEV_ADMIN_PASSWORD`` for dev only).

### `FacetDescriptor`

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | `layer`, `head`, `expert`, `component`, `all`, … |
| `label` | string | Human label |
| `description` | string | |
| `coverage` | number | Fraction of keys with a value for this facet |
| `assigned_count` | integer | |
| `unassigned_count` | integer | |
| `cardinality` | integer | Distinct facet values |
| `sample_values` | string[] | Up to 8 examples |
| `offered` | boolean | UI should offer if true |

### `SuggestedFacetPair`

| Field | Type | Description |
|-------|------|-------------|
| `row_facet` | string | Facet id or `leaf` |
| `col_facet` | string | |
| `description` | string | |
| `cell_count` | integer | Estimated matrix cells |
| `fits_budget` | boolean | `cell_count <= max_cells` |

### `FacetAnalysisResponse`

**`GET /api/runs/{run_id}/facet-analysis`** → `200` body.

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | string | Resolved regex used for analysis |
| `metric` | string | Echo |
| `path_pattern` | string | Echo |
| `resolved_pattern` | string | Same as resolved full-key pattern |
| `match_count` | integer | Catalog matches (all dtypes) |
| `plottable_match_count` | integer | `float`/`int` only |
| `facets` | `FacetDescriptor[]` | Offered facets only |
| `suggested_pairs` | `SuggestedFacetPair[]` | |
| `error` | string | Present only on invalid empty scope (handler usually returns `400` instead) |

Query: `pattern` **or** `metric` (+ optional `path_pattern`); `max_cells` (default `512`, alias `max_keys`).

### `MatrixCell`

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Metric key for this cell |
| `row` | integer | Row index |
| `col` | integer | Column index |
| `value` | number \| null | Scalar at requested `step` (nearest step) |

### `MatrixLayoutResponse`

**`GET /api/runs/{run_id}/matrix-layout`** → `200` body.

Core layout (from `propose_layout` / aggregation helpers):

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `matrix` or `tiles` |
| `confidence` | number | Heuristic layout confidence |
| `row` | object \| null | `{ "facet": string, "labels": string[] }` |
| `col` | object \| null | Same shape as `row` |
| `cells` | `MatrixCell[]` | Matrix mode |
| `tiles` | object[] | Tile mode (`key`, optional `value`, …) |

Additional fields merged by `build_matrix_layout_payload`:

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | string | Resolved regex |
| `metric`, `path_pattern` | string | Echo |
| `match_count` | integer | Catalog match count |
| `plottable_match_count` | integer | |
| `layout_keys_planned` | integer | Keys after scope filter |
| `leaf_count` | integer | |
| `layout_truncated` | boolean | |
| `max_cells` | integer | Response budget |
| `step` | number | Step used for values |
| `depth` | string | Depth used (`auto`, `leaf`, …) |
| `depth_requested` | string | |
| `depth_adjusted` | boolean | |
| `scope` | string | Substring drill scope |
| `suggested_depths` | array | Depth hints for UI |
| `children_available` | boolean | More keys than returned cells |
| `steps` | integer[] | Sample of logged steps (up to 500) |

Query parameters: same scope as facet-analysis, plus `step` (default `0`), `max_cells`,
`depth`, `agg` (default `max`), `scope`, `row_facet`, `col_facet`, `layout_mode`
(default `auto`).

---

## Operations

### List runs

`GET /api/runs`

**Response `200`:** `RunSummary[]` (JSON array).

Side effect: ingests any discovered run missing `run.db`.

---

### Run manifest

`GET /api/runs/{run_id}/manifest`

**Response `200`:** `RunManifest`

**Response `404`:** `Error` — run not found (ingest attempted once).

---

### Metric catalog

`GET /api/runs/{run_id}/catalog`

**Query parameters:**

| Name | Type | Description |
|------|------|-------------|
| `limit` | integer | Max rows in `metrics`; `match_count` still full |

**Response `200`:** `CatalogResponse`

**Response `400`:** if `pattern`, `metric`, or `path_pattern` is set — use **`GET …/metrics/search`**.

---

### Metric search

`GET /api/runs/{run_id}/metrics/search`

**Query parameters:**

| Name | Type | Description |
|------|------|-------------|
| `pattern` | string | Regex on full metric key |
| `metric` | string | Scoped family `section/metric` (e.g. `train/loss`, `val/loss`) |
| `path_pattern` | string | Regex on suffix path |
| `limit` | integer | Max rows in `metrics`; `match_count` still full |
| `summary` | `0` \| `1` | When `1`, build a collapsed path tree for all matches (`summary` field); default `limit` is treated as `0` for row payload |

**Response `200`:** `MetricsSearchResponse` — includes **`summary`** when `summary=1` (see
`MetricSearchSummary` in OpenAPI).

**Response `400`:** missing scope (`pattern` or `metric`).

Invalid regex for a provided filter → `metrics: []`, `match_count: 0`, `truncated: false`.

---

### Metric families

`GET /api/runs/{run_id}/metric-families`

**Response `200`:** `MetricFamiliesResponse`

---

### Single series

`GET /api/runs/{run_id}/series`

**Query parameters:**

| Name | Required | Default | Description |
|------|----------|---------|-------------|
| `key` | yes | — | Metric key |
| `max_points` | no | `2000` | Downsample cap |
| `x_axis` | no | `step` | `step`, `global_step`, `epoch`, `logged_at` |

**Response `200`:** `SeriesResponse`

**Response `400`:** missing `key`

---

### Series by prefix

`GET /api/runs/{run_id}/series/prefix`

**Query parameters:**

| Name | Required | Default |
|------|----------|---------|
| `prefix` | yes | — |
| `max_points` | no | `2000` |

**Response `200`:** `SeriesPrefixResponse`

---

### Series batch

`POST /api/runs/{run_id}/series/batch`

**Request body:** `SeriesBatchRequest`

**Response `200`:** `SeriesBatchResponse`

**Response `422`:** missing or empty `keys`

---

### Facet analysis

`GET /api/runs/{run_id}/facet-analysis`

**Query:** scope (`pattern` or `metric` + `path_pattern`), `max_cells` (optional).

**Response `200`:** `FacetAnalysisResponse`

**Response `400`:** missing scope

---

### Matrix layout

`GET /api/runs/{run_id}/matrix-layout`

**Query:** scope, `step`, `max_cells`, `depth`, `agg`, `scope`, `row_facet`, `col_facet`, `layout_mode`.

**Response `200`:** `MatrixLayoutResponse`

**Response `400`:** missing scope

---

### Workspace (user layout / panel preferences)

`GET /api/workspace`

**Response `200`:** `WorkspaceDocumentResponse` — `sections` may be `[]` on first launch.

`PUT /api/workspace`

**Request body:** `WorkspaceDocumentRequest` (replaces the entire document).

**Response `200`:** `WorkspaceDocumentResponse` echoing saved `sections`.

Implementation: `backend/bev/store/account_store.py` + `workspace_store.py`. Legacy
`{store_dir}/workspace.db` is imported once into the accounts DB when a root has no saved config.

**Browser-only (not in SQLite):** run visibility checkboxes and focused-run selection are kept in
`localStorage` (`webui/src/lib/runGroups.js`). On first load, if the server document is empty but
`localStorage` has a v2 workspace, the UI imports it once via `PUT /api/workspace`. The UI also
mirrors sections to `localStorage` on change as a offline fallback.

---

### Static UI

`GET /`

**Response `200`:** `index.html` and assets when built.

**Response `503`:**

```json
{
  "error": "UI not built",
  "hint": "cd tutorials/pathologies/bev/webui && npm install && npm run build"
}
```

---

## Storage layout

| Path | Purpose |
|------|---------|
| `~/.bev/accounts.db` (or `BEV_ACCOUNTS_DB`) | Users, scan roots, per-user workspace configs |
| `<store_dir>/runs/<wandb_run_id>/run.db` | One SQLite database per ingested W&B run |
| `<store_dir>/workspace.db` | **Legacy** single-document store (migrated on read) |
| W&B scan root (`--root`) | Offline `run-*.wandb` files; not part of the API store |

`store_version` in `run_meta` increments when ingest applies **new** history events (not on
no-op re-ingest at EOF); the UI uses it to refresh series and catalog caches.

---

## How the UI uses these endpoints

### Workspace bootstrap

1. **`GET /api/workspace`** on mount — hydrate sections and panels.
2. Debounced **`PUT /api/workspace`** when the user edits layout or panel settings.
3. Multipanel charts load series only after a committed selection (keys stored on the panel in
   this document); see **Multipanel committed load** below.

### Sidebar metrics (focused run)

1. **`GET /api/runs/{run_id}/manifest`** — polling / `store_version`.
2. **`GET /api/runs/{run_id}/metric-families`** — always. One row per scoped **`section/metric`**
   family with a plottable key count (e.g. `train/update_rank`, `val/loss`).

Click: `plottable_count === 1` → line panel; otherwise multipanel for that family
(`webui/src/lib/sidebarMetrics.js`).

### Line panel picker (settings)

Full **`GET /catalog`** only when `metric_count ≤ 800` (panel settings tree).

### Panel metric dropdown (heatmap / multipanel)

**`GET /api/runs/{run_id}/metric-families`** (same as sidebar).

### Panel regex preview

- Settings path field: **`GET …/metrics/search`** with `limit=0` (match count only, no row payload).
- **Pattern exploration** modal: same endpoint with **`summary=1`** for the collapsed tree.
- **`GET /facet-analysis`** for **heatmap** facet hints in panel settings (when the modal is closed).

### Multipanel committed load

1. **`GET …/metrics/search`** (Apply selection) — resolve keys up to panel cap.
2. Persist keys on the panel; on display/reload use **`POST …/series/batch`** with `{ keys }` only.
   UI groups charts by **metric path hierarchy** (`buildMultipanelGroupTree`), not server facets.
   Committed selection stays valid across backend restarts and **`store_version`** bumps from new
   training steps; re-apply only when metric scope / regex / drill scope change (fingerprint mismatch).

---

## SQLite schema (`accounts.db`)

Default path **`~/.bev/accounts.db`** (`BEV_ACCOUNTS_DB` override). Schema version **1**
(`backend/bev/store/account_store.py`).

### `users`

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | INTEGER PK | |
| `username` | TEXT UNIQUE | e.g. `admin` |
| `password_hash` | TEXT | PBKDF2-SHA256 (`pbkdf2_sha256$…`); never plain text |
| `role` | TEXT | `admin` / `user` (future RBAC) |
| `created_at` | TEXT | |

Bootstrap: on first open, **`admin`** is created if missing (password from `BEV_ADMIN_PASSWORD`,
default **`password`** — dev only).

### `scan_roots`

| Column | Type | Notes |
|--------|------|-------|
| `scan_root_id` | INTEGER PK | |
| `root_path` | TEXT UNIQUE | Resolved absolute `bev serve --root` |
| `first_opened_at` | TEXT | Set on insert |
| `last_opened_at` | TEXT | Updated each serve / touch |

### `user_workspace_configs`

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | INTEGER FK → `users` | |
| `scan_root_id` | INTEGER FK → `scan_roots` | |
| `sections_json` | TEXT | Panel layout JSON |
| `updated_at` | TEXT | |

Primary key **`(user_id, scan_root_id)`**.

### Legacy `workspace.db` (under `store_dir`)

Pre-accounts installs used a single row table `workspace_document`. On load, if no row exists
for the current admin + scan root, that file is read once and copied into `user_workspace_configs`.

---

## SQLite schema (per-run `run.db`)

`SCHEMA_VERSION = 2` (`backend/bev/store/schema.py`). Foreign keys should be enabled on
connections.

### `schema_migrations`

| Column | Type | Notes |
|--------|------|-------|
| `version` | INTEGER PK | Applied schema version |
| `applied_at` | TEXT | Default `datetime('now')` |

### `run_meta` (single row, `id = 1`)

| Column | Type | Notes |
|--------|------|-------|
| `wandb_run_id` | TEXT UNIQUE | Run id |
| `display_name`, `project`, `entity` | TEXT | From W&B |
| `wandb_dir`, `wandb_file` | TEXT | Source paths |
| `checkpoint_dir` | TEXT | Optional |
| `store_version` | INTEGER | Bumped on ingest (UI cache key) |
| `created_at`, `updated_at` | TEXT | |

### `ingest_state` (single row, `id = 1`)

| Column | Type | Notes |
|--------|------|-------|
| `byte_offset` | INTEGER | Resume offset in `.wandb` file |
| `file_size`, `file_mtime_ns` | INTEGER | Change detection |
| `last_wandb_step` | INTEGER | |
| `last_ingest_at` | TEXT | |

### `steps`

| Column | Type | Notes |
|--------|------|-------|
| `step` | INTEGER PK | Training step |
| `global_step`, `epoch`, `logged_at` | INTEGER / REAL | Optional axes |

Index: `idx_steps_global_step`.

### `metrics_family`

| Column | Type | Notes |
|--------|------|-------|
| `family_id` | INTEGER PK | |
| `name` | TEXT UNIQUE | Segment after `train/` (e.g. `update_rank`) |
| `plottable_count` | INTEGER | Count of `float`/`int` keys in catalog (maintained at ingest) |

Populated when metrics are ingested; **`GET /metric-families`** reads this table only.

### `metrics_catalog`

| Column | Type | Notes |
|--------|------|-------|
| `metric_id` | INTEGER PK | |
| `key` | TEXT UNIQUE | Full W&B metric key |
| `dtype` | TEXT | `float`, `int`, `string`, `null` |
| `first_step`, `last_step` | INTEGER | |
| `n_points` | INTEGER | |
| `path_segments` | TEXT | JSON array (ingest metadata) |
| `storage`, `storage_ref` | TEXT | `inline` vs `file` (future) |
| `family_id` | INTEGER FK | → `metrics_family` |

Indexes: `idx_metrics_catalog_key`, `idx_metrics_catalog_family_id`.

### `metric_values`

| Column | Type | Notes |
|--------|------|-------|
| `metric_id`, `step` | PK, FK → `metrics_catalog`, `steps` | |
| `value_real`, `value_int`, `value_text` | | One value type per row |

Indexes: `idx_metric_values_step`.

### `alarm_events` (placeholder)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `step`, `rule_id`, `severity`, `message`, `metric_key`, `fired_at` | | Not wired in UI yet |

### `snapshot_index` (placeholder)

| Column | Type | Notes |
|--------|------|-------|
| `snapshot_step` | INTEGER PK | Checkpoint heatmaps (future) |
| `checkpoint_label`, `manifest_path`, `indexed_at` | TEXT | |

---

## Related docs

- Product roadmap: `PLAN.md`
- Metric matrix design: `docs/heatmap-matrix-design.md`
