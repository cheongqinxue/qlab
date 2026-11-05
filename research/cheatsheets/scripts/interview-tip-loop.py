#!/usr/bin/env python3
"""Emit AGENT_LOOP_TICK every 30s with a random interview tip for chat delivery."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

TIPS_PATH = Path(__file__).resolve().parent.parent / "data" / "interview-tips.jsonl"
SENTINEL = "AGENT_LOOP_TICK_INTERVIEW_TIP"
INTERVAL_S = 60


def load_tips() -> list[dict[str, str]]:
    lines = TIPS_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def emit_tick(tip: dict[str, str]) -> None:
    text = tip["text"]
    source = tip.get("source", "")
    prompt = (
        "Interview tip loop: post ONE tip to the user in this chat only. "
        "No desktop notifications. One or two complete sentences; include the substance below. "
        "Do not add filler or ask if they want more. "
        f"Tip: {text}"
    )
    if source:
        prompt += f" (from {source})"
    payload = json.dumps({"prompt": prompt}, ensure_ascii=False)
    print(f"{SENTINEL} {payload}", flush=True)


def main() -> None:
    tips = load_tips()
    if not tips:
        print("No tips loaded", file=sys.stderr)
        sys.exit(1)
    while True:
        time.sleep(INTERVAL_S)
        emit_tick(random.choice(tips))


if __name__ == "__main__":
    main()
