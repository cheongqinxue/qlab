# Adam instability (`adam_collapse`)

## In this tutorial

We set **Adam `epsilon` extremely small** (`1e-16`), **high LR**, **no clipping**, and **no warmup**. Watch for NaNs, loss spikes, erratic `train/adam_second_moment/layer.*`, and **`train/adam_max_r` / `train/adam_max_g2_over_v`** (effective Adam step and SPAM-style spike scores; `train/adam_low_moment_frac` is fragile surface area, not a standalone alarm).

## What goes wrong

Adam’s per-parameter step is roughly \(\eta \cdot \hat m / (\sqrt{\hat v} + \epsilon)\). When \(\hat v\) is mis-estimated, decoupled from current gradients, or polluted by rare spikes, effective preconditioning can push optimization into an **unstable** regime—loss spikes or divergence even when SGD might survive.

## How practitioners spot it

- **Sudden loss spikes** mid-training (common in LLM pretraining logs).
- **Adam second moments** \(v_t\): some coordinates with near-zero \(v\) (huge steps) or stale \(v\) after a quiet period then rising \(|g|\).
- **Gradient spike detectors:** flag coordinates where \(g_i^2 / v_i\) exceeds a threshold (SPAM, ICLR 2025).
- **Theory:** preconditioned Hessian eigenvalues exceeding \(2/\eta\) (Bai et al., 2025).

## Literature: detection and response

### Provable non-convergence of Adam

**Reddi et al. (2018)** — *On the Convergence of Adam and Beyond* — give **counterexamples** where Adam diverges; **spot** failure via constructed stochastic gradients. Propose **AMSGrad** (max over past \(v_t\)) so \(\sqrt{v_t}\) does not decrease spuriously.

**Fix (optimizer math):** enforce non-decreasing second-moment envelope: \(\hat v_t \leftarrow \max(\hat v_{t-1}, v_t)\) before the denominator.

### LLM-scale instability

**Zhang et al. (2023)** — *A Theory on Adam Instability in Large-Scale Machine Learning* — observe **divergent** LLM training when update vectors have large norm and weak correlation with true descent direction; link to **large batch** and **diminishing gradient components** in Adam’s ratio.

**Detection:** track alignment between Adam update and stochastic gradient; monitor update norm vs loss improvement.

### Loss spikes from second-moment decay

**Bai et al. (2025)** — *Adaptive Preconditioners Trigger Loss Spikes in Adam* — identify **decoupling**: when \(|g_t| \ll \sqrt{v_t}\), \(v_t\) decays under \(\beta_2\) and stops tracking \(g_t^2\); preconditioned curvature can exceed stability threshold \(2/\eta\).

**Detection:** gradient-directional curvature \(\lambda_{\mathrm{grad}} = g^\top \hat H g / \|g\|^2\) crossing \(2/\eta\).

**Fixes (conceptual):** reset or clip moments after spikes; increase \(\epsilon\); gradient spike nulling (SPAM); AdamW with tuned \(\beta_2\); switch to SGD+momentum in fragile phases.

### Gradient spikes harm LLM training

**SPAM (ICLR 2025)** — detect spikes via \(g_i^2/v_i\), **zero or clip** spiked coordinates; report lower perplexity when spikes are removed.

## References

- Reddi, Kale, Kumar (2018). [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237).
- Zhang et al. (2023). [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871).
- Bai et al. (2025). [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://arxiv.org/abs/2506.04805).
- SPAM (2025). [Gradient Spike Detection for LLM Training](https://proceedings.iclr.cc/paper_files/paper/2025/file/7a70ad3d9c704fb9b81b5c69eda722dc-Paper-Conference.pdf).
