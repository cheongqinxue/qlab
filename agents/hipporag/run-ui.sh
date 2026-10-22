#!/usr/bin/env bash
# Start the HippoRAG Streamlit tester (logs on this terminal).
#
# Prerequisites:
#   Agent running (./hipporag/run-agent.sh or docker compose up)
#   agents/.env
#
# Usage:
#   ./agents/hipporag/run-ui.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_common.sh
source "${SCRIPT_DIR}/_run_common.sh"

for arg in "$@"; do
  case "${arg}" in
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

load_hipporag_env
UI_PORT="${HIPPORAG_UI_PORT:-8502}"

if ! curl -sf "${HIPPORAG_AGENT_URL}/health" >/dev/null 2>&1; then
  echo "Warning: agent not healthy at ${HIPPORAG_AGENT_URL}/health — start ./hipporag/run-agent.sh first." >&2
fi

export PYTHONPATH="${AGENTS_DIR}:${HIPPORAG_DIR}"
export HIPPORAG_AGENT_URL

echo "Streamlit UI: http://127.0.0.1:${UI_PORT}"
echo "Agent logs (other terminal): cd ${AGENTS_DIR} && docker compose -f hipporag/docker-compose.yml logs -f agent"
echo "Ctrl+C stops the UI only."
echo "---"

cd "${REPO_ROOT}"
exec uv run streamlit run "${HIPPORAG_DIR}/ui/app.py" \
  --server.port "${UI_PORT}" \
  --server.address 127.0.0.1
