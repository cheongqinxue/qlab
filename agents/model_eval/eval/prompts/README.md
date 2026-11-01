# Curated eval prompts

Committed, repeatable prompt slices for fair model comparison (`run_suite.py`).

| File | Count | Source |
|------|-------|--------|
| `fixture.jsonl` | 3 | Offline smoke fixture |
| `tool_use.jsonl` | 7 | Harness-adapted multi-tool (Hermes-inspired) |
| `instruction_following.jsonl` | 6 | `google/IFEval` |
| `hallucination.jsonl` | 6 | `truthfulqa/truthful_qa` |
| `sycophancy.jsonl` | 5 | Hand-authored false-premise / flattery bait |
| `reasoning.jsonl` | 6 | `allenai/ai2_arc` (`ARC-Challenge`) |
| `math.jsonl` | 8 | `EleutherAI/hendrycks_math` (`algebra`) |
| `coding.jsonl` | 6 | `openai/openai_humaneval` |

Run a dimension:

```bash
./agents/model_eval/run_eval.sh --prompts agents/model_eval/eval/prompts/math.jsonl --model qwen3.5:4b
```

Each row includes `curation_note` and benchmark `metadata` for failure-mode probes.
