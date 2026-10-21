# Committed offline W&B scan roots

Playwright and manual UI work read from these directories (no runtime `wandb.init`).

| Directory | Purpose |
|-----------|---------|
| `e2e_minimal/` | Small run (`train/loss`, `attention_max_weight`); use `npm run test:minimal` in `e2e/` |
| `llama70b_update_rank_scan/` | **Default Playwright + dev UI** (~11k `update_rank` keys; same as `run-heatmap-dev.sh`) |

## Regenerate

From repo root:

```bash
PYTHONPATH=tutorials/pathologies/bev/backend uv run python -m bev.fixtures e2e_minimal
PYTHONPATH=tutorials/pathologies/bev/backend uv run python -m bev.fixtures llama70b_update_rank_scan --checkpoints 1
```

## Serve (heatmap / matrix dev)

```bash
cd /path/to/qlab
./tutorials/pathologies/bev/run-heatmap-dev.sh
```

Or manually:

```bash
BEV_SCAN_ROOT=tutorials/pathologies/bev/backend/bev/fixtures/data/llama70b_update_rank_scan \
  PYTHONPATH=tutorials/pathologies/bev/backend uv run python -m bev serve --root "$BEV_SCAN_ROOT"
```

Override Playwright fixture: `BEV_SCAN_ROOT` or `BEV_E2E_SCAN_ROOT`.
