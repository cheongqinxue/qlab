#!/usr/bin/env bash
# Run eval for 8B-tier queue items 2–8 (item 1 qwen3_5_4B done separately).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="${ROOT}/agents/tmp/candidates_8b_batch.log"
EVAL="${ROOT}/agents/model_eval/run_eval.sh"
RESULTS="${ROOT}/agents/model_eval/eval/results"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== remaining 8B queue $(date -Iseconds) ==="

write_yaml() {
  local dir=$1 model_name=$2 params=$3 tag=$4 quant=$5 modality=$6
  cat >"${RESULTS}/${dir}/ollama.yaml" <<EOF
# ${model_name} — 8B-tier candidate eval
model_name: ${model_name}
params: ${params}
modality: ${modality}

ollama:
  tag: ${tag}
  quantization: ${quant}

  options:
    temperature: 0.1
    top_p: 0.95
    num_ctx: 32768
    num_predict: 32768
EOF
}

run_model() {
  local dir=$1 tag=$2
  echo "=== ${dir} pull ${tag} $(date -Iseconds) ==="
  if ! ollama pull "${tag}"; then
    echo "=== PULL FAILED ${dir} ==="
    return 1
  fi
  echo "=== ${dir} eval $(date -Iseconds) ==="
  if ! "${EVAL}" --model-dir "${RESULTS}/${dir}/"; then
    echo "=== EVAL FAILED ${dir} ==="
    return 1
  fi
  echo "=== ${dir} OK $(date -Iseconds) ==="
  return 0
}

FAILED=0

# 2 — OpenBMB 8B (MiniCPM5-8B text not on Ollama; use MiniCPM-V 4.5 8B stack)
mkdir -p "${RESULTS}/minicpm5_8B"
write_yaml minicpm5_8B "MiniCPM-V 4.5 (8B)" "8B" "openbmb/minicpm-v4.5:8b" "Q4_K_M" "T+I->T"
run_model minicpm5_8B "openbmb/minicpm-v4.5:8b" || FAILED=$((FAILED + 1))

mkdir -p "${RESULTS}/lfm2_5_8B"
write_yaml lfm2_5_8B "LFM2.5-8B-A1B" "8B MoE" "lfm2.5:8b" "Q4_K_M" "T->T"
run_model lfm2_5_8B "lfm2.5:8b" || FAILED=$((FAILED + 1))

mkdir -p "${RESULTS}/granite4_2_8B"
write_yaml granite4_2_8B "Granite 4.2-8B" "8B" "granite4.2:8b" "Q4_K_M" "T->T"
run_model granite4_2_8B "granite4.2:8b" || FAILED=$((FAILED + 1))

mkdir -p "${RESULTS}/gemma4_e4b"
write_yaml gemma4_e4b "Gemma 4 E4B" "~4.5B eff." "gemma4:e4b" "Q4_K_M" "T+I->T"
run_model gemma4_e4b "gemma4:e4b" || FAILED=$((FAILED + 1))

mkdir -p "${RESULTS}/zaya1_8B"
write_yaml zaya1_8B "ZAYA1-8B" "8.4B MoE" "zaya1:8b" "Q4_K_M" "T->T"
run_model zaya1_8B "zaya1:8b" || FAILED=$((FAILED + 1))

mkdir -p "${RESULTS}/bonsai_2_4B"
if ollama pull "bonsai-2:4b-q2_0"; then
  write_yaml bonsai_2_4B "Ternary Bonsai 2 4B" "4B" "bonsai-2:4b-q2_0" "Q2_0_ternary" "T+I->T"
  run_model bonsai_2_4B "bonsai-2:4b-q2_0" || FAILED=$((FAILED + 1))
else
  write_yaml bonsai_2_4B "Ternary Bonsai 2 4B" "4B" "bonsai-2:4b" "Q2_0_ternary" "T+I->T"
  run_model bonsai_2_4B "bonsai-2:4b" || FAILED=$((FAILED + 1))
fi

mkdir -p "${RESULTS}/llama3_1_8B"
write_yaml llama3_1_8B "Llama 3.1 8B Instruct" "8B" "llama3.1:8b" "Q4_K_M" "T->T"
run_model llama3_1_8B "llama3.1:8b" || FAILED=$((FAILED + 1))

echo "=== rescore 8B batch $(date -Iseconds) ==="
(cd "${ROOT}" && uv run python agents/model_eval/scripts/rescore_qualitative.py) || true
(cd "${ROOT}" && uv run python agents/model_eval/scripts/export_cheatsheet_data.py) || true
echo "=== batch end $(date -Iseconds) failures=${FAILED} ==="

echo "=== queued: LFM3-1.7B-Base eval $(date -Iseconds) ==="
"${ROOT}/agents/model_eval/scripts/run_lfm3_1_7B_base_eval.sh" || FAILED=$((FAILED + 1))

echo "=== all queued jobs $(date -Iseconds) failures=${FAILED} ==="
exit 0
