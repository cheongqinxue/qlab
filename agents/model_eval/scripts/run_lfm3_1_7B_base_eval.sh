#!/usr/bin/env bash
# Re-eval lfm3_1_7B folder using LiquidAI/LFM3-1.7B-Base (not post-trained lfm3:1.7b).
# Queued after 8B candidate batch — see run_candidates_8b_remaining.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="${ROOT}/agents/tmp/lfm3_1_7B_base_eval.log"
RESULTS="${ROOT}/agents/model_eval/eval/results/lfm3_1_7B_base"
EVAL="${ROOT}/agents/model_eval/run_eval.sh"

mkdir -p "${RESULTS}" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== LFM3-1.7B-Base eval $(date -Iseconds) ==="

# Official Base GGUF when published; else community quant from LiquidAI/LFM3-1.7B-Base.
OLLAMA_TAG="${LFM3_BASE_OLLAMA_TAG:-hf.co/LiquidAI/LFM3-1.7B-Base-GGUF:Q4_K_M}"
QUANT="${LFM3_BASE_QUANT:-Q4_K_M}"

cat >"${RESULTS}/ollama.yaml" <<EOF
# LFM3-1.7B-Base (pre-trained checkpoint — not chat/post-trained LFM3-1.7B)
# Publisher: Liquid AI (https://huggingface.co/LiquidAI/LFM3-1.7B-Base)
# GGUF: official repo when available; override tag via LFM3_BASE_OLLAMA_TAG.

model_name: LFM3-1.7B-Base
params: 1.7B
modality: T->T
thinking_model: false

ollama:
  tag: ${OLLAMA_TAG}
  hf_base_repo: LiquidAI/LFM3-1.7B-Base
  hf_gguf_repo: LiquidAI/LFM3-1.7B-Base-GGUF
  quantization: ${QUANT}

  options:
    temperature: 0.1
    top_k: 50
    repeat_penalty: 1.1
    num_ctx: 32768
    num_predict: 32768
EOF

echo "=== pull ${OLLAMA_TAG} ==="
if ! ollama pull "${OLLAMA_TAG}"; then
  FALLBACK="hf.co/VelociQ8/LFM3-1.7B-Base-Q8A64-GGUF"
  echo "=== primary pull failed; fallback ${FALLBACK} ==="
  OLLAMA_TAG="${FALLBACK}"
  QUANT="Q8A64"
  sed -i "s|tag:.*|tag: ${OLLAMA_TAG}|" "${RESULTS}/ollama.yaml"
  sed -i "s|quantization:.*|quantization: ${QUANT}|" "${RESULTS}/ollama.yaml"
  ollama pull "${OLLAMA_TAG}" || { echo "=== PULL FAILED ==="; exit 1; }
fi

echo "=== eval $(date -Iseconds) ==="
"${EVAL}" --model-dir "${RESULTS}/" || exit 1

echo "=== rescore $(date -Iseconds) ==="
(cd "${ROOT}" && uv run python agents/model_eval/scripts/rescore_qualitative.py) || true
(cd "${ROOT}" && uv run python agents/model_eval/scripts/export_cheatsheet_data.py) || true
echo "=== done $(date -Iseconds) ==="
