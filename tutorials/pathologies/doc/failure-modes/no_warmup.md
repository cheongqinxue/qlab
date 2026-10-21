# No learning-rate warmup (`no_warmup`)

**Encoder MLM only** in this tutorial.

## In this tutorial

We use a **moderately high LR** (`5e-4`) with **`warmup_ratio=0`** on Post-LN BERT. Compare to baseline (`5e-5`, 10% warmup). Expect rougher early loss, possible spikes, and slower or unstable convergence vs baseline—not always catastrophic on tiny BERT, but the signature is “bad early phase.”

## What goes wrong

A **full learning rate from step 0** interacts badly with **large gradients at initialization** in Post-LN transformers. Early updates overshoot; Adam’s moments are cold-started; layer norms have not settled.

## How practitioners spot it

- **Loss curve:** spike or plateau in first hundreds/thousands of steps; need to restart with warmup.
- **Gradient norms:** elevated in early steps (Xiong et al. theory for output-adjacent layers).
- **Hyperparameter sweeps:** warmup length and peak LR are highly coupled in transformer pretraining recipes (GPT, BERT, T5).

## Literature: detection and response

### Why warmup exists for Post-LN

**Xiong et al. (2020)** — prove at init, **Post-LN** has **large expected gradients** near the output; full \(\eta\) is unstable. **Warmup** linearly ramps \(\eta_t\) from 0 to \(\eta_{\max}\).

**Fix (schedule):** \(\eta_t = \eta_{\max} \cdot \min(1, t/T_{\mathrm{warmup}})\) (or cosine after warmup).

### Pre-LN alternative

Same paper: **Pre-LN** (\(x + \mathrm{Sublayer}(\mathrm{LN}(x))\)) has **well-behaved** gradients at init → can **remove** warmup in their experiments.

**Fix (architecture):** move LN inside residual; add final LN before logits.

### Large LM practice

**Kaplan et al. (2020)** scaling laws; **GPT-3** and follow-on recipes use warmup + cosine decay. **Detection** is empirical: without warmup, large models diverge immediately.

## References

- Xiong et al. (2020). [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html).
- Kaplan et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
