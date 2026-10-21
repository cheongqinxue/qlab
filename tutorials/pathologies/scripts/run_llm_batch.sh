#!/usr/bin/env bash
# Run all LLM pathology experiments sequentially; emit RUN_COMPLETE lines for monitoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATHOLOGIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PATHOLOGIES_DIR}/../.." && pwd)"
RUN_SH="${PATHOLOGIES_DIR}/run.sh"
SUMMARY_PY="${SCRIPT_DIR}/summarize_wandb_run.py"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

MODES=(baseline explode vanish adam_collapse logit_collapse)
EXPERIMENTS=(llm_sft llm_grpo)

for experiment in "${EXPERIMENTS[@]}"; do
    for mode in "${MODES[@]}"; do
        run_id="${experiment}_${mode}"
        echo "========== START ${run_id} $(date -Is) =========="
        "${RUN_SH}" "${experiment}" "${mode}"
        uv run python "${SUMMARY_PY}" --run-name "${run_id}" || true
        echo "RUN_COMPLETE: ${run_id} $(date -Is)"
        echo ""
    done
done

echo "BATCH_COMPLETE $(date -Is)"
