# Metric matrix / heatmap design (BEV)

Status: **proposal + test fixture** (attention-specific heatmap in the UI is legacy; replace with layout engine below).

## Problem

Today’s attention heatmap assumes keys like `train/attention_max_weight/.../layer.N/.../head.H`.
That breaks for `train/update_rank/bert.encoder.layer.12.attention.self.query.weight`, activation RMS,
FFN blocks, norms, etc.

We need a **metric-family panel** that:

1. Takes a **regex or prefix** (user filter), not a hard-coded parser.
2. Asks the **server** to infer structure from logged keys (`path_segments` + stats).
3. Falls back to a **tile strip** when no stable 2D facet exists.
4. Supports **resolution** (group/aggregate) for large transformers (24 layers × many components × heads).
5. Uses **hover tooltips** (full key + value at step), not fragile axis labels alone.

## How other tools handle this

| Product / pattern | What they do | Scale limit |
|-------------------|--------------|-------------|
| **W&B workspaces** | Regex multi-metric **line** panels; sections mirror key hierarchy; paginated grids of panels; resize syncs panel size in a section. No true layer×head matrix for arbitrary keys. | Hundreds of keys → pagination, not one heatmap. |
| **W&B histograms** | Distribution **heatmaps over time** (bins × step), not layer×module grids. | Different problem (time × bin). |
| **TensorBoard** | Scalars by tag; TB histograms as time heatmaps; no first-class module tree matrix. | Tag explosion → many charts. |
| **Seaborn / trellis** | **FacetGrid**: pick `row` / `col` / `hue` dimensions; small multiples share axes. | Needs explicit facet columns. |
| **Neuron / CNN viewers** | Layer slider + channel grid; hooks per module; **one layer at a time** or feature-map grid. | “Don’t show 100+ layers at once” (DNN Viewer). |
| **Genome / matrix browsers** | **Virtualized** 2D canvas, zoom levels, aggregation at coarse LOD. | Best analogy for 24×12×components. |

**Takeaway:** Nobody plots an entire transformer’s per-parameter scalars in one static matrix. They combine **(a) hierarchy navigation**, **(b) small multiples**, **(c) virtualization + LOD**, and **(d) explicit facet choice**.

## Options for BEV (pick one or combine)

### Option A — Facet picker + matrix (recommended core)

- Server scans matched keys and returns **candidate facets** (e.g. `layer`, `head`, `component`, literal segments with high cardinality).
- User or auto-default picks **row facet** and **column facet** (like FacetGrid).
- Cells = one scalar at selected step (or sparkline on click).
- **Pros:** Works for `update_rank`, `effective_rank`, attention heads, FFN, norms.  
- **Cons:** User may need to pick facets once per metric family.

### Option B — Auto matrix + tile fallback (your spec)

- Server runs **consensus facet detection** on slash/dot tokens (see `matrix_layout.propose_layout`).
- If two orthogonal facets explain ≥80% of keys → **matrix** with row/col headers from token values.
- Else → **tile strip**: fixed square tiles, wrap at **80% of panel inner width**, no row/col headers; identity via tooltip only.
- **Pros:** Zero config for regular naming (`layer.N`, `head.N`, module paths).  
- **Cons:** Ambiguous trees (same key depth varies) need tie-break rules; must document confidence score in API.

### Option C — Multipanel only (status quo+)

- Keep regex **multipanel** line charts; heatmap panel becomes “matrix mode” only when layout API returns `mode=matrix`.
- **Pros:** Already scales with pagination.  
- **Cons:** Not a single glance heatmap; many charts for 24 layers.

### Option D — Virtualized canvas + LOD (large models)

- One panel, **zoom levels**: L0 = layer aggregates, L1 = layer×component, L2 = full params / heads.
- Render only visible tiles; cell size clamped (3–30px) with window resize.
- **Pros:** Handles full BERT-24 + 12 heads + components.  
- **Cons:** Most engineering; needs aggregation rules per metric type (mean/max of children).

### Option E — Table / pivot (W&B Tables style)

- Pivot UI: rows = module path, columns = step or metric variant.  
- **Pros:** Familiar for wide data.  
- **Cons:** Less “heatmap” feel; wide tables hurt performance.

## Recommended path

1. **Short term:** Implement **Option B API** on the server + **Option A** overrides in panel settings (row/col dropdown fed by API).
2. **UI:** Single **Metric matrix** panel type (rename from “Attention heatmap”): regex filter, step slider, layout mode `matrix | tiles`, tooltip on hover, glow on cell.
3. **Sizing:** `max-width: 80%` of panel content; `max-height: min(70vh, panel)` with resize observer; tile size derived from count and caps (reuse 3–30px clamp).
4. **Medium term:** Add **Option D** LOD when cell count &gt; threshold (e.g. 400).

## API sketch (server)

```
GET /api/runs/{id}/matrix-layout?pattern=...&step=...&max_cells=512
→ {
  "mode": "matrix" | "tiles",
  "match_count": <catalog regex matches>,
  "plottable_match_count": <float/int matches>,
  "layout_keys_planned": <keys used for layout inference (all plottable matches)>,
  "layout_truncated": false,          // layout no longer capped by max_cells
  "values_truncated": true|false,     // response cells/tiles capped by max_cells
  "layout_cell_count" | "layout_tile_count": <planned bucket count>,
  "max_cells": 512,
  "confidence": 0.0-1.0,
  "row": { "facet": "layer", "labels": ["0","1",...] },
  "col": { "facet": "component", "labels": ["attention.self", ...] },
  "cells": [{ "key", "row", "col", "value" }],
  "tiles": [{ "key", "value" }]   // when mode=tiles
}
```

`max_keys` is a deprecated alias for `max_cells` (response value budget only).

**Phase 2 (LOD):** query params `depth` (`auto` | `0` | `1` | `2` | `leaf`), `agg` (`max` | `mean` | `min` | `p95`), optional `scope`, optional `row_facet` / `col_facet`. Response adds `leaf_count`, `depth`, `depth_adjusted`, `aggregated`, `agg`, `suggested_depths`, `children_available`, and per-cell `leaf_count` (keys rolled into each bucket).

Layer indices are parsed from generic module paths (`layer.N`, `layers.N`, `blocks.N`, …), not model-specific tables.

Layout logic lives in Python (`bev/catalog/matrix_layout.py`), tested against the large fixture catalog.

## Test fixture

See `bev/fixtures/wandb_large_module_tree.py` and `tests/test_large_module_tree_fixture.py`.

Simulates a **24-layer, 12-head** encoder (~1.1k scalar keys) with per-module metrics:

- `train/update_rank/bert.encoder.layer.{L}.{component}.{param}`
- `train/effective_rank/...`
- `train/activation_rms/bert.encoder.layer.{L}.{component}`
- `train/attention_max_weight/bert.encoder.layer.{L}.attention.self/head.{H}`

Use **small** `num_layers=4`, `num_heads=4` in unit tests; full size for manual `bev serve` benchmarks.

```bash
uv run pytest tutorials/pathologies/bev/backend/tests/test_large_module_tree_fixture.py -q
```

Manual UI iteration (full 24×12 catalog):

```bash
uv run python tutorials/pathologies/bev/scripts/bootstrap_large_fixture.py /tmp/bev-large-scan
uv run python -m bev serve --root /tmp/bev-large-scan
```

Try regex filters such as `train/update_rank/`, `train/attention_max_weight/`, `train/effective_rank/`.

### Llama 70B-scale stress run (`update_rank` only)

Head-sliced attention + per-layer FFN + embed + `lm_head`:

- **11,762** metric keys per checkpoint (80 layers, 64 Q heads, 8 KV heads, GQA)
- **20** checkpoint steps by default → **235,240** scalar events in one offline run

```bash
./tutorials/pathologies/bev/run-heatmap-dev.sh
```

(`scripts/ensure-fixtures.sh` regenerates data if missing; store defaults to `~/.bev/store-heatmap-dev`.)

Filter example: `train/update_rank/model\.layers\.\d+\.self_attn\.q_proj\.weight/head\.\d+`

Pytest: `test_llama70b_scale_fixture.py` (full ingest marked `@pytest.mark.slow`).
