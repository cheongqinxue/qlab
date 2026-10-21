#!/usr/bin/env bash
# Start the BEV web UI for a pathology lab run (or any .data/<RUN_ID>/wandb tree).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATHOLOGIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PATHOLOGIES_DIR}/.data"

DEFAULT_RUN_ID="${BEV_RUN_ID:-encoder_mlm_repeat_shift_adam_collapse}"
STORE_DIR="${BEV_STORE_DIR:-${HOME}/.bev/store}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [RUN_ID] [serve options...]

Opens the BEV UI for W&B logs under:
  tutorials/pathologies/.data/<RUN_ID>/wandb/

Defaults:
  RUN_ID     ${DEFAULT_RUN_ID}  (override: BEV_RUN_ID or first argument)
  Store      ${STORE_DIR}  (override: BEV_STORE_DIR or --store-dir)

Examples:
  $(basename "$0")
  $(basename "$0") encoder_mlm_explode
  $(basename "$0") encoder_mlm_repeat_shift_adam_collapse --port 8765
  BEV_RUN_ID=bilstm_vanish $(basename "$0")

Any extra flags are passed to \`bev serve\` (e.g. --port, --no-watch — see serve --help).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ID="${DEFAULT_RUN_ID}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  RUN_ID="$1"
  shift
fi

WANDB_ROOT="${DATA_DIR}/${RUN_ID}/wandb"
if [[ ! -d "${WANDB_ROOT}" ]]; then
  echo "error: W&B directory not found: ${WANDB_ROOT}" >&2
  echo "Train a run first, e.g.: tutorials/pathologies/run.sh encoder_mlm baseline" >&2
  exit 1
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

ARGS=(--root "${WANDB_ROOT}" --store-dir "${STORE_DIR}")
[[ "${WATCH}" -eq 1 ]] && ARGS+=(--watch)
[[ "${OPEN}" -eq 1 ]] && ARGS+=(--open)
ARGS+=("${EXTRA[@]}")

echo "BEV run: ${RUN_ID}"
echo "W&B root: ${WANDB_ROOT}"
echo "Store: ${STORE_DIR}"

exec "${SCRIPT_DIR}/scripts/serve.sh" "${ARGS[@]}"
