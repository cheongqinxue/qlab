#!/usr/bin/env bash
# Serve the exported second-moment viewer (index.html + data.json) over HTTP.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_DIR="${1:-${SCRIPT_DIR}/../.data/encoder_mlm_repeat_shift_adam_collapse/second_moment_viewer}"
PORT="${PORT:-8765}"

if [[ ! -f "${VIEWER_DIR}/index.html" ]]; then
  echo "Missing ${VIEWER_DIR}/index.html — run build_second_moment_viewer.py first." >&2
  exit 1
fi
if [[ ! -f "${VIEWER_DIR}/data.json" ]]; then
  echo "Missing ${VIEWER_DIR}/data.json — run build_second_moment_viewer.py first." >&2
  exit 1
fi

echo "Serving ${VIEWER_DIR} at http://127.0.0.1:${PORT}/"
cd "${VIEWER_DIR}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
