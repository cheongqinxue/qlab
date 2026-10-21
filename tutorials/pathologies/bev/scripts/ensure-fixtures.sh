#!/usr/bin/env bash
# Regenerate committed offline W&B fixtures if run files are missing.
set -euo pipefail

BEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QLAB_ROOT="$(cd "$BEV_ROOT/../../.." && pwd)"
export PYTHONPATH="${BEV_ROOT}/backend${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR="${BEV_ROOT}/backend/bev/fixtures/data"

need_regen() {
  local scan_root="$1"
  [[ ! -d "${scan_root}" ]] && return 0
  find "${scan_root}" -name 'run-*.wandb' -print -quit | grep -q . || return 0
  return 1
}

regen() {
  local name="$1"
  shift
  echo "==> regenerating fixture: ${name}" >&2
  cd "${QLAB_ROOT}"
  uv run python -m bev.fixtures "${name}" "$@"
}

if need_regen "${DATA_DIR}/e2e_minimal"; then
  regen e2e_minimal
else
  echo "==> ok: e2e_minimal" >&2
fi

if need_regen "${DATA_DIR}/llama70b_update_rank_scan"; then
  regen llama70b_update_rank_scan --checkpoints 1
else
  echo "==> ok: llama70b_update_rank_scan" >&2
fi
