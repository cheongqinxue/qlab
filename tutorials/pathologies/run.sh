#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TRAIN="${SCRIPT_DIR}/train.py"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

usage() {
    cat <<EOF
Usage:
  $(basename "$0") <experiment> [failure_mode]
  $(basename "$0") ablation_transformer [variant] [failure_mode]

Experiments:
  encoder_mlm  encoder_mlm_repeat_shift  llm_sft  llm_grpo  bilstm  ablation_transformer

Failure modes:
  baseline
  Training pathologies: explode vanish adam_collapse attention_sink rank_collapse
                        logit_collapse layer_grad_imbalance representation_collapse
  Best-practice violations (encoder_mlm): no_warmup decay_norm_bias bad_mlm_mask

Ablation variants: standard  no_scale  no_norm  no_residual

Examples:
  $(basename "$0") encoder_mlm explode
  $(basename "$0") ablation_transformer no_scale baseline

W&B: logs under tutorials/pathologies/.data/<run>/wandb/
  Default WANDB_MODE=offline when no API key / local server is configured.
  Local server: wandb server start  then  export WANDB_BASE_URL=http://localhost:8080
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

EXPERIMENT="$1"
FAILURE_MODE="baseline"
VARIANT="standard"
VARIANT_ARGS=()

if [[ "${EXPERIMENT}" == "ablation_transformer" ]]; then
    VARIANT="${2:-standard}"
    FAILURE_MODE="${3:-baseline}"
    VARIANT_ARGS=(--variant "${VARIANT}")
else
    FAILURE_MODE="${2:-baseline}"
fi

PYTHONPATH="${REPO_ROOT}/tutorials" "${PYTHON}" -c "
import pathologies.experiments
from pathologies.experiment import get_experiment
get_experiment('${EXPERIMENT}')
" || {
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    exit 1
}

if [[ "${EXPERIMENT}" == "ablation_transformer" ]]; then
    RUN_ID="${EXPERIMENT}_${VARIANT}_${FAILURE_MODE}"
else
    RUN_ID="${EXPERIMENT}_${FAILURE_MODE}"
fi

DATA_DIR="${SCRIPT_DIR}/.data/${RUN_ID}"
CKPT_DIR="${DATA_DIR}/checkpoints"
WANDB_DIR="${DATA_DIR}/wandb"

mkdir -p "${CKPT_DIR}" "${WANDB_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-pathologies}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_RUN_NAME="${RUN_ID}"
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_BASE_URL:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

echo "W&B project: ${WANDB_PROJECT}  run: ${RUN_ID}  mode: ${WANDB_MODE:-online}"
echo "W&B dir: ${WANDB_DIR}"

"${PYTHON}" "${TRAIN}" \
    -E "${EXPERIMENT}" \
    -F "${FAILURE_MODE}" \
    -C "${CKPT_DIR}" \
    -L "${WANDB_DIR}" \
    --run-name "${RUN_ID}" \
    ${VARIANT_ARGS[@]+"${VARIANT_ARGS[@]}"}

if [[ "${WANDB_MODE:-}" == "offline" ]]; then
    echo "Done. Sync offline runs with: wandb sync ${WANDB_DIR}"
else
    echo "Done. View run: ${WANDB_BASE_URL:-https://wandb.ai}/${WANDB_ENTITY:-}/${WANDB_PROJECT}/${RUN_ID}"
    echo "Workspace layout (once): uv run python ${SCRIPT_DIR}/scripts/setup_wandb_workspace.py"
fi
