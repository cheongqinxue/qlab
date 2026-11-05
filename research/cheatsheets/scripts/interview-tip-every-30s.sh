#!/usr/bin/env bash
# Chat loop: wakes the agent every 60s to post a random tip here (not notify-send).
# Tips pool: ../data/interview-tips.jsonl (779+ lines from cheatsheets).
# Stop: pkill -f interview-tip-loop.py

set -euo pipefail
exec python3 "$(dirname "$0")/interview-tip-loop.py"
