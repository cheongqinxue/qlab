# Logit / output collapse (`logit_collapse`)

## In this tutorial

We use **high learning rate**, **no warmup**, **no weight decay**—settings that push the model toward **overconfident** softmax outputs. Monitor `train/logit_entropy` (vocab entropy; on LLM SFT aliased from TRL `entropy`) **decreasing** vs baseline.

## What goes wrong

The distribution \(p = \mathrm{softmax}(z)\) becomes **too peaked**: entropy \(H(p) = -\sum_k p_k \log p_k\) falls. The model may still lower training loss while **calibration** and **generalization** suffer; wrong classes receive near-zero probability (Szegedy et al.).

## How practitioners spot it

- **Entropy of predictions** on train/val (our `logit_entropy`).
- **Max logit gap:** \(z_y - \max_{k\neq y} z_k\) grows large (label smoothing paper’s motivation).
- **Calibration metrics:** ECE, reliability diagrams—overconfidence without label smoothing.
- **LLMs:** repetitive or dull generations when policy entropy collapses during RL (related but distinct from SFT).

## Literature: detection and response

### Label smoothing

**Szegedy et al. (2016)** — *Rethinking the Inception Architecture* — **spot** overfitting via logits driven to \(\infty\) on the true class. **Fix:** replace one-hot \(q\) with \((1-\varepsilon)\delta_y + \varepsilon u\).

**Math:** cross-entropy with soft targets penalizes extreme logits; equivalent to adding \(\mathrm{KL}(u \| p)\) toward a uniform prior.

### Confidence penalty

**Pereyra et al. (2017)** — penalize **low entropy** outputs directly (confidence penalty) or use label smoothing; connect smoothing to **KL** direction vs entropy maximization.

**Fix (loss):** \( \mathcal{L} + \beta H(p)\) (encourage higher entropy).

### Neural collapse and logit structure

**Papyan et al. (2020)** — collapse of features and classifiers can yield **simplex** geometry in logits; useful for classification but can interact badly with calibration if unchecked.

### LLM SFT / RL

High LR without warmup can sharpen next-token distributions quickly. **Mitigations:** label smoothing (some instruction-tuning recipes), **temperature** at inference, **KL to reference** in RLHF, entropy bonuses in RL (not our SFT path).

**Detection in LLMs:** monitor **token-level entropy** during training (TRL’s `entropy` metric); compare to `logit_entropy` in our lab.

## References

- Szegedy et al. (2016). [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
- Pereyra et al. (2017). [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548).
- Papyan et al. (2020). [Prevalence of neural collapse](https://arxiv.org/abs/2006.07739).
