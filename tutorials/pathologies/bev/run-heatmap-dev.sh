#!/usr/bin/env bash
# BEV UI for heatmap / metric-matrix work (Llama-scale update_rank fixture).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAN_ROOT="${SCRIPT_DIR}/backend/bev/fixtures/data/llama70b_update_rank_scan"
STORE_DIR="${BEV_HEATMAP_STORE_DIR:-${HOME}/.bev/store-heatmap-dev}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [serve options...]

Starts BEV against the committed Llama 70B-scale fixture:
  ${SCAN_ROOT}

~11,762 train/update_rank/* keys (1 checkpoint). Use an Attention heatmap panel
with regex e.g. train/update_rank/model\\.layers\\.\\d+\\.self_attn\\.q_proj\\.weight/head\\.\\d+

Defaults:
  Store  ${STORE_DIR}  (override: BEV_HEATMAP_STORE_DIR or --store-dir)

Ensures fixtures exist (runs scripts/ensure-fixtures.sh) and builds static/ if needed.
Extra flags go to \`bev serve\` (--port, --no-watch, --no-open, etc.).

Examples:
  $(basename "$0")
  $(basename "$0") --port 8765 --no-open
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

"${SCRIPT_DIR}/scripts/ensure-fixtures.sh"

need_ui_build=0
if [[ ! -f "${SCRIPT_DIR}/static/index.html" ]]; then
  need_ui_build=1
elif find "${SCRIPT_DIR}/webui/src" -type f -newer "${SCRIPT_DIR}/static/index.html" -print -quit | grep -q .; then
  need_ui_build=1
fi
if [[ "${need_ui_build}" -eq 1 ]]; then
  echo "Building BEV UI (static/)…" >&2
  (cd "${SCRIPT_DIR}/webui" && npm install && npm run build)
fi

WATCH=1
OPEN=1
EXTRA=()
for arg in "$@"; do
  case "${arg}" in
    --no-watch) WATCH=0 ;;
    --no-open) OPEN=0 ;;
    *) EXTRA+=("${arg}") ;;
  esac
done

ARGS=(--root "${SCAN_ROOT}" --store-dir "${STORE_DIR}")
[[ "${WATCH}" -eq 1 ]] && ARGS+=(--watch)
[[ "${OPEN}" -eq 1 ]] && ARGS+=(--open)
ARGS+=("${EXTRA[@]}")

echo "BEV heatmap dev"
echo "Scan root: ${SCAN_ROOT}"
echo "Store: ${STORE_DIR}"
echo "Suggested heatmap regex (until matrix UI lands):"
echo "  train/update_rank/model\\.layers\\.0\\.self_attn\\.q_proj\\.weight/head\\.\\d+"
echo "(Sidebar metric tree is omitted above 800 keys; avoid broad multipanel regexes.)"

exec "${SCRIPT_DIR}/scripts/serve.sh" "${ARGS[@]}"
