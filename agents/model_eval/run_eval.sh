#!/usr/bin/env bash
# Model eval harness — Langfuse-tagged ReAct runs over curated JSONL.
#
#   ./agents/model_eval/run_eval.sh --model-dir agents/model_eval/eval/results/lfm2_5_vl_450M

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
# shellcheck source=scripts/_lib.sh
source "${AGENTS_ROOT}/scripts/_lib.sh"
load_agents_env
export AGENT_NAME="model_eval"
# On host, map docker-internal langfuse-web:3000 to localhost:${LANGFUSE_UI_PORT:-3000}
if [[ "${LANGFUSE_BASE_URL:-}" == "http://langfuse-web:3000" ]]; then
  export LANGFUSE_BASE_URL="http://localhost:${LANGFUSE_UI_PORT:-3000}"
fi
export PYTHONPATH="${AGENTS_ROOT}:${EVAL_DIR}:${PYTHONPATH:-}"

exec uv run python "${EVAL_DIR}/scripts/run_suite.py" "$@"
