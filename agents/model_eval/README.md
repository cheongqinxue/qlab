# Model eval harness

Fair **React/direct** comparison across quantized models, plus **failure-mode probes** (not full benchmark reproduction). Eval runs **fail fast**: harness errors and LLM/API failures propagate (no per-case exception swallowing).

## Goals

1. **Performance comparison** — **tool_use** cases use the shared LangGraph ReAct agent with eval tools; all other dimensions use a **single-turn chat invoke** (user prompt only, no tool schemas). Same curated prompts per dimension.
2. **Failure probing** — deterministic probes for **tool_use**, **math**, **coding**, and **reasoning** (e.g. `missing_expected_tool`, answer key mismatch). **coding** runs HumanEval `check()` in a subprocess after AST safety validation (no destructive imports/calls). **instruction_following**, **hallucination**, and **sycophancy** are scored from committed human review (`eval/human_qualitative_review.jsonl`); probes only flag empty answers and runtime errors.

## `model_eval_core` layout

| Package | Role |
|---------|------|
| `domain/` | `EvalDimension`, pass/fail (`scoring`), cheatsheet labels |
| `data/` | Repo paths, JSON/JSONL I/O, curated `prompts` |
| `harness/` | `run_case`, ReAct agent, eval tools |
| `probes/` | Case failure modes + infrastructure tags |
| `coding/` | HumanEval extract/exec, AST safety, rescoring |
| `qualitative/` | Human review lookup + offline rubrics |
| `metrics/` | GPU/VRAM runtime + token throughput |
| `text/` | Strip chain-of-thought wrappers (`strip_thinking`) |

Import from the package you need, e.g. `from model_eval_core.harness import run_case`.

## Prompts

Curated JSONL only — see **[CURATION.md](CURATION.md)**. HF streaming and selection live under **`agents/tmp/model_eval_curation/`** (agent scratch, gitignored). Promote stable sets into **`eval/prompts/`**.

## Scratch vs committed

| Path | Role |
|------|------|
| `eval/prompts/*.jsonl` | Committed, repeatable curated prompt sets per dimension + `fixture.jsonl` |
| `agents/tmp/` | Gitignored scratch (HF browse scripts, eval reports, local model lists) |
| `models.sample.yaml` | Example Ollama quant tags |

## Langfuse

- `AGENT_NAME=model_eval` (default in `run_eval.sh`)
- Optional `LANGFUSE_TAGS=quant-eval,local`
- Scores: `eval_pass`, `failure_mode_count`, categorical `failure_modes`

## Run

```bash
./agents/model_eval/run_eval.sh \
  --prompts agents/model_eval/eval/prompts/fixture.jsonl \
  --model qwen3.5:4b \
  --limit 2
```

Reports default to `agents/tmp/eval_reports/`.

**Re-score coding (no LLM):** HumanEval exec on saved `final_answer`:

```bash
uv run python agents/model_eval/scripts/rescore_coding.py
uv run python agents/model_eval/scripts/rescore_coding.py --model-dir agents/model_eval/eval/results/lfm3_1_7B
```

## Cheatsheet matrix

After updating qualitative human scores in `eval/human_qualitative_review.jsonl`, refresh stored passes and summaries:

```bash
uv run python agents/model_eval/scripts/rescore_qualitative.py
uv run python agents/model_eval/scripts/export_cheatsheet_data.py
```

**Infrastructure tags (no LLM):** thinking budget, repetition loops, and related tool-stack modes on saved JSONL (also applied automatically during `run_suite` / live probes). Backfill existing results:

```bash
uv run python agents/model_eval/scripts/rescore_infrastructure.py
uv run python agents/model_eval/scripts/rescore_infrastructure.py --model-dir agents/model_eval/eval/results/lfm3_1_7B
```

Writes `failure_modes` / `signals.infrastructure` on each row and `infrastructure_failure_counts` on `summary.json`. Langfuse traces get `infrastructure_categories` and `max_completion_tokens` scores when tracing is on.

**Throughput (tok/s):** `summary.json` → `token_stats` sums `usage` on AI messages in `responses.jsonl`. **Out tok/s** = completion tokens ÷ suite `runtime.wall_s` (else sum of case `latency_s`). New runs populate this via `run_suite`; backfill existing folders with `backfill_metrics.py` (tokens only, no VRAM probe) then re-export.

Open **`research/cheatsheets/agents-model-eval.html`** (sortable dimension table + row click for prompts/outputs).

Writes `research/cheatsheets/js/agents-model-eval-data.js` (sortable dimension table + row click for prompts/outputs).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest agents/model_eval/tests -q
```
