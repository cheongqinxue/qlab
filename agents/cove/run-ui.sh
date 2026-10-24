#!/usr/bin/env bash
# Start the CoVe Streamlit tester (logs on this terminal).
#
# Prerequisites:
#   Agent running (./cove/run-agent.sh or docker compose up)
#   agents/.env
#
# Usage:
#   ./agents/cove/run-ui.sh

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

load_cove_env
COVE_UI_PORT="${COVE_UI_PORT:-8503}"

cd "${REPO_ROOT}"
exec uv run streamlit run agents/cove/ui/app.py \
  --server.port "${COVE_UI_PORT}" \
  --server.address 127.0.0.1
