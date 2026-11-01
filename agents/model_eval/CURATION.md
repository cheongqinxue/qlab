# Prompt curation (agent workflow)

The **eval harness does not auto-curate** from Hugging Face. Curated prompts are committed JSONL files under `agents/model_eval/eval/prompts/` consumed by `run_suite.py`.

| Location | Role |
|----------|------|
| `agents/tmp/model_eval_curation/` | **Gitignored** — HF streaming browse/append scripts for the agent |
| `agents/model_eval/eval/prompts/` | **Committed** — repeatable prompt sets per dimension (`math.jsonl`, `tool_use.jsonl`, etc.) |

Workflow: agent runs `browse_hf.py` in `agents/tmp/model_eval_curation/` → picks indices → `append_prompt.py` writes directly into `agents/model_eval/eval/prompts/<dimension>.jsonl`.
