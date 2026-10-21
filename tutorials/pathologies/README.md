# Training pathologies

Learn to recognize **unhealthy training** before it wastes compute — not by chasing eval
accuracy alone, but by watching gradients, updates, activations, attention, and weight rank
in **Weights & Biases**.

This folder runs real HuggingFace models through **Trainer** / **TRL** with deliberately bad
hyperparameters so failure modes produce obvious chart signatures. The full diagnostic toolkit
from the first iteration lives in [`../telemetry_deprecated/`](../telemetry_deprecated/) and
is documented here as the target surface area.

## Quick start

```bash
# One experiment name + failure mode (all pathologies on the same entrypoint)
./tutorials/pathologies/run.sh encoder_mlm baseline
./tutorials/pathologies/run.sh encoder_mlm explode
./tutorials/pathologies/run.sh ablation_transformer no_scale baseline
```

Or directly:

```bash
export WANDB_MODE=offline
export WANDB_PROJECT=pathologies

uv run python tutorials/pathologies/train.py \
  -E encoder_mlm \
  -F explode \
  -C /tmp/ckpt \
  -L /tmp/wandb \
  --run-name encoder_mlm_explode \
  --max-steps 100
```

Outputs land under `tutorials/pathologies/.data/<run>/`:
- `checkpoints/` — Trainer output
- `wandb/` — W&B logs (offline by default; sync with `wandb sync`)

**Local W&B server** (no cloud account):

```bash
pip install wandb
wandb server start          # Docker; UI at http://localhost:8080
export WANDB_BASE_URL=http://localhost:8080
wandb login --host=http://localhost:8080
./tutorials/pathologies/run.sh encoder_mlm baseline
```

Custom metrics appear as `train/logit_entropy`, `train/update_rank/...`,
`train/adam_second_moment/layer.0`, etc. For LLM SFT/GRPO, `logit_entropy` matches
TRL’s `train/entropy` (chunked CE skips `lm_head.forward`).

### BEV (local diagnostics UI, in development)

Work-in-progress replacement for opinionated W&B views: local `.wandb` ingest, shared-axis
panels, metric tree picker, checkpoint heatmaps, and alarms. Self-contained package in
[`bev/`](bev/) (Python in `bev/backend/bev`, tests in `bev/backend/tests/`). Track work in
[`bev/PLAN.md`](bev/PLAN.md) (`[ ]` / `[x]` tasks). Run tests:
`uv run pytest tutorials/pathologies/bev/backend/tests`. UI: [`bev/run.sh`](bev/run.sh).

### W&B workspace layout (manual)

Training (`run.sh` / `train.py`) does **not** push workspace layout. Use
[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces) only when you change
panels or sections:

```bash
# once after clone, or after editing wandb_workspace.py
uv run python tutorials/pathologies/scripts/setup_wandb_workspace.py

# optional overrides (avoid --new-view unless you want another duplicate tab)
uv run python tutorials/pathologies/scripts/setup_wandb_workspace.py \
  --entity your-team --project pathologies
```

This creates or updates the **Pathology lab** view: Core training, Gradients,
Updates, Attention, Outputs, Rank, Adam, Activations. Per-layer grad/Adam panels
are split into layer bands with distinct colors.

Layout source: `pathologies/instrumentation/wandb_workspace.py`.

Remove duplicate saved views (W&B GraphQL `deleteView`; not in `wandb-workspaces` SDK):

```bash
uv run python tutorials/pathologies/scripts/cleanup_wandb_workspace_views.py \
  --display-name "Pathology lab" --dry-run
# then without --dry-run; add --keep-latest to leave one tab
```

### Hook target map

Regenerate `hook_targets.yaml` (patterns → concrete module and parameter names per experiment):

```bash
uv run python tutorials/pathologies/scripts/verify_hook_targets.py
# fast path without HF downloads: --skip-hf
```

First run downloads models into `<repo>/.cache/` (`train.py` overrides broken global HF cache
paths). See `.env-template` for `WANDB_*` and HF cache overrides.

## Models and experiments

| Family | HuggingFace model | Training stack | Dataset |
|--------|-------------------|----------------|---------|
| **Encoder** | `google/bert_uncased_L-2_H-128_A-2` | `Trainer` (MLM) | WikiText-2 |
| **LLM** | `HuggingFaceTB/SmolLM2-135M-Instruct` | `SFTTrainer` / `GRPOTrainer` | SmolTalk / GSM8K |
| **BiLSTM** | Custom `PreTrainedModel` | `Trainer` (classification) | AG News |
| **Ablation encoder** | Custom mini-transformer | `Trainer` (classification) | Synthetic |

### Registered experiments (6 total)

```
encoder_mlm  encoder_mlm_repeat_shift  llm_sft  llm_grpo  bilstm  ablation_transformer
```

**`baseline`** — control run with sane hyperparameters (not a pathology).

Pathologies are split into two buckets in `experiments/_common.py`:

| Bucket | When it shows up | Modes |
|--------|------------------|-------|
| **Training pathologies** | Can happen even with reasonable intent | `explode`, `vanish`, `adam_collapse`, `attention_sink`, `rank_collapse`, `logit_collapse`, `layer_grad_imbalance`, `representation_collapse` |
| **Best-practice violations** | Known recipe mistakes for encoder pretraining | `no_warmup`, `decay_norm_bias`, `bad_mlm_mask` *(encoder MLM experiments)* |

Each profile sets a **`min_steps`** budget; runs use `max(--max-steps, min_steps)`.

### `encoder_mlm_repeat_shift` (data lever)

Same **`--failure-mode`** presets as `encoder_mlm`; only the **batch schedule** changes:

- **No shuffling** — `SequentialSampler` over a synthetic dataset.
- Steps **`1 .. 99`** — identical batch (fixed WikiText passage, fixed MLM masks).
- Step **`100`** — one batch from a different passage (shift).
- Steps **`101 .. 200`** — repeat that shift batch (see whether metrics settle or stay volatile).

Use this when random WikiText makes a pathology hard to see (e.g. Adam moment staleness before a shift). Example:

```bash
./run.sh encoder_mlm_repeat_shift adam_collapse
```

## Pluggable telemetry

```python
from pathologies.instrumentation import TelemetryConfig, build_trainer_callbacks

callbacks, collector = build_trainer_callbacks(
    model,
    TelemetryConfig.encoder_mlm(),
)
trainer = Trainer(..., callbacks=callbacks)
```

Presets: `minimal()`, `encoder_mlm()`, `ablation_transformer()`, `bilstm()`.
CLI override: `--telemetry none|minimal|encoder|ablation|bilstm`.

## Diagnostic panel

| Diagnostic | W&B metric (prefix `train/`) | What it catches |
|------------|------------------------------|-----------------|
| Global grad L2 | `grad_norm` | Overall gradient health |
| Per-layer grad L2 | `grad_norm/layer.N` | Depth-wise vanish/explode |
| Update-to-weight ratio | `update_to_weight_ratio`, `update_ratio/*` | Updates too large or too small |
| **Update rank (ΔW)** | `update_rank/{param}` | Low-rank or collapsed **updates** (not just weights) |
| Activation RMS | `activation_rms/{module}` | Activation blow-up |
| Logit entropy | `logit_entropy` | Output collapse / overconfidence |
| Attention entropy | `attention_entropy/{module}` | Attention collapse |
| Attention max weight | `attention_max_weight/{module}` | Peaked attention |
| **Per-head imbalance** | `attention_head_imbalance/{module}`, `attention_max_weight/{module}/head.N` | One head dominates |
| Weight effective rank | `effective_rank/{param}` | Rank collapse in Q/K weights |
| **Adam second moment** | `adam_second_moment/layer.N` | `exp_avg_sq` per layer (Adam instability) |
| **Adam health** | `adam_max_r`, `adam_p99_r`, `adam_low_moment_frac`, `adam_max_g2_over_v`, `adam_frac_r_above_threshold` | Fragile \(v\) + effective step / SPAM spike (see `adam_collapse` doc) |
| Embedding anisotropy | `embedding_anisotropy` | Representation collapse |

### Effective rank vs update rank

Both use the **participation ratio** on singular values:

\[
\text{participation ratio} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}
\]

- **`effective_rank/{param}`** — rank of the **weight matrix** \(W\)
- **`update_rank/{param}`** — rank of the **update** \(\Delta W\) after each step

Rank collapse in weights is slow; rank collapse in updates can show up earlier.

### Per-head attention imbalance

When attention weights are 4D `(batch, heads, seq, seq)` (BERT with `attn_implementation="eager"`):

- **`attention_head_imbalance/{layer}`** — ratio of max to min per-head peak weight (1.0 = balanced; high = one head hogs mass)
- **`attention_max_weight/{layer}/head.N`** — per-head peak attention

### Adam second moment

After each optimizer step, mean RMS of `sqrt(exp_avg_sq)` grouped by encoder layer.
Watch for layers where the second moment vanishes (`adam_collapse`) or spikes.

---

## Failure modes (spot in W&B)

Compare each mode against **`baseline`** on the same experiment.

| Mode | Signature |
|------|-----------|
| `explode` | `grad_norm` ↑, `update_to_weight_ratio` ↑, loss spike |
| `vanish` | Flat loss, tiny `update_to_weight_ratio` |
| `adam_collapse` | NaN / spike; `adam_second_moment/layer.*` erratic |
| `attention_sink` | `attention_max_weight` ↑, `attention_head_imbalance` ↑ |
| `rank_collapse` | `effective_rank/{Q,K}` → ~1 |
| `logit_collapse` | `logit_entropy` ↓ |
| `layer_grad_imbalance` | `update_ratio/embedding` & `update_ratio/head` ≫ `update_ratio/encoder` |

---

## Suggested lab workflow

1. Run **baseline**; note typical `grad_norm` and loss slope.
2. Run a pathology; overlay runs in W&B (same project, filter by run name).
3. Zoom early steps for `explode`; use `update_rank` vs `effective_rank` for rank pathologies.

## Project layout

```
pathologies/
  README.md
  train.py            ← CLI entrypoint
  run.sh              ← W&B env + experiment runner
  instrumentation/    ← callbacks, hooks, diagnostics
  experiments/        ← encoder, llm, bilstm + failure presets
  tests/
```

## Dependencies

From repo root: `torch`, `transformers`, `trl`, `accelerate`, `datasets`, `wandb`,
`wandb-workspaces`, `python-dotenv`.
