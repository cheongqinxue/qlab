#!/usr/bin/env bash
# One-shot local test: shared services (if needed) → HippoRAG agent → Streamlit UI.
#
# Usage:
#   ./agents/hipporag/run.sh
#   ./agents/hipporag/run.sh --no-langfuse
#   ./agents/hipporag/run.sh --no-ollama --agent-only
#   ./agents/hipporag/run.sh --detach   # agent detached; UI in foreground (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_LANGFUSE=1
START_OLLAMA=1
START_UI=1
EXTRA_AGENT_ARGS=(--detach)

for arg in "$@"; do
  case "${arg}" in
    --no-langfuse) START_LANGFUSE=0 ;;
    --no-ollama) START_OLLAMA=0 ;;
    --agent-only) START_UI=0 ;;
    --foreground) EXTRA_AGENT_ARGS=(--foreground) ;;
    --detach | -d) EXTRA_AGENT_ARGS=(--detach) ;;
    -h | --help)
      sed -n '2,10p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${START_LANGFUSE}" -eq 1 ]]; then
  "${AGENTS_DIR}/scripts/start-langfuse.sh"
fi

if [[ "${START_OLLAMA}" -eq 1 ]]; then
  "${AGENTS_DIR}/scripts/start-ollama.sh"
fi

"${SCRIPT_DIR}/run-agent.sh" "${EXTRA_AGENT_ARGS[@]}"

if [[ "${START_UI}" -eq 1 ]]; then
  exec "${SCRIPT_DIR}/run-ui.sh"
fi

echo "Agent running. Start UI later: ${SCRIPT_DIR}/run-ui.sh"
