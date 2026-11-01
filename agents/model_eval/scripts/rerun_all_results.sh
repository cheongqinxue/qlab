#!/usr/bin/env bash
# Rerun full 44-case suite for every folder under eval/results/ with ollama.yaml.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS="${ROOT}/agents/model_eval/eval/results"
LOG="${ROOT}/agents/tmp/rerun_eval_direct_invoke.log"
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
FAILED=0
for dir in "$RESULTS"/*/; do
  [[ -f "${dir}ollama.yaml" ]] || continue
  name="$(basename "$dir")"
  echo "=== ${name} $(date -Iseconds) ===" | tee -a "$LOG"
  if ! "${ROOT}/agents/model_eval/run_eval.sh" --model-dir "$dir" 2>&1 | tee -a "$LOG"; then
    echo "=== FAILED ${name} (suite aborted mid-run) ===" | tee -a "$LOG"
    FAILED=$((FAILED + 1))
  fi
done
echo "=== rescore qualitative $(date -Iseconds) ===" | tee -a "$LOG"
(cd "$ROOT" && uv run python agents/model_eval/scripts/rescore_qualitative.py) 2>&1 | tee -a "$LOG"
(cd "$ROOT" && uv run python agents/model_eval/scripts/export_cheatsheet_data.py) 2>&1 | tee -a "$LOG"
echo "Done $(date -Iseconds); model_failures=${FAILED}" | tee -a "$LOG"
exit 0
