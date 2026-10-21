# Rank / spectral collapse (`rank_collapse`)

## In this tutorial

We apply **strong weight decay** on attention projections while training normally otherwise. Watch `train/effective_rank/*` on Q/K weights (participation ratio of singular values) drift toward **~1**—weights become effectively rank-one.

## What goes wrong

Weight matrices \(W\) lose usable singular values: \(\sigma_1 \gg \sigma_2 \approx \cdots \approx 0\). The model has fewer degrees of freedom in its linear maps; attention or MLP layers behave like **low-rank bottlenecks**. Related but distinct from **neural collapse** (feature geometry at end of training) and **dimensional collapse** in contrastive learning (subspace collapse of embeddings).

## How practitioners spot it

- **Spectrum of weights or updates:** plot singular values of \(W\) or \(\Delta W\); participation ratio \((\sum \sigma_i)^2 / \sum \sigma_i^2\) drops.
- **Our telemetry:** `effective_rank/{param}` each step; `update_rank/{param}` for low-rank **updates** even when \(W\) still looks full-rank.
- **Contrastive SSL:** covariance of embeddings has fewer significant eigenvalues (Jing et al.).

## Literature: detection and response

### Dimensional collapse in contrastive learning

**Jing et al. (2022)** — *Understanding Dimensional Collapse in Contrastive Self-supervised Learning* — show embeddings span a **lower-dimensional subspace** than the ambient dimension even with negative pairs. **Detect** via eigenvalue spectrum of representation covariance; **explain** via augmentation variance vs weight decay dynamics.

**Fix:** DirectCLR (directly optimize representation subspace); stronger augmentations; stop-gradient / projector designs (SimCLR family).

### Neural collapse (supervised)

**Papyan et al. (2020)** — *Prevalence of neural collapse* — late training: class features collapse to means, means form a **simplex ETF**, classifier aligns with means. **Detection:** UMAP of penultimate features; cosine structure between class means.

**Not identical to our rank_collapse mode** (we attack **weight** rank via L2), but both reduce effective dimensionality—worth contrasting in W&B.

### Controlling collapse for OOD vs generalization

**Harun et al. (2025, ICML)** — stronger neural collapse helps **OOD detection** but hurts **transfer**; use **entropy regularization** to weaken collapse for generalization, or **fixed Simplex ETF projector** to enforce it for detection.

**Fix (loss):** add terms that control spread of singular values or feature norms.

### Weight decay as implicit low-rank pressure

Heavy **L2** on matrices shrinks all singular values; without countervailing gradient signal, **effective rank** drops. **AdamW** decouples decay from adaptive steps but does not remove this effect.

**Mitigation:** lower decay on attention projections; **LoRA** intentionally train low-rank adapters; spectral norm constraints (less common in LLMs).

## References

- Jing et al. (2022). [Understanding Dimensional Collapse in Contrastive SSL](https://arxiv.org/abs/2110.09348).
- Papyan, Han, Donoho (2020). [Prevalence of neural collapse during training](https://arxiv.org/abs/2006.07739).
- Harun et al. (2025). [Controlling Neural Collapse Enhances OOD Detection and Transfer Learning](https://proceedings.mlr.press/v267/harun25a.html).
