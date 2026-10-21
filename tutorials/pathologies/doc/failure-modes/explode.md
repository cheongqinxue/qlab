# Gradient explosion (`explode`)

## In this tutorial

We induce explosion with a **very high learning rate**, **no gradient clipping** (`max_grad_norm` effectively disabled), **no warmup**, and **no weight decay**. Expect rising `train/grad_norm`, large `train/update_to_weight_ratio`, loss spikes or NaNs, and unstable Adam moments.

## What goes wrong

Updates grow until activations or logits blow up; backward passes produce enormous gradients. In RNNs this was the classic “exploding gradient” alongside vanishing; in deep nets it appears as sudden loss spikes, overflow, or divergence.

## How practitioners spot it

- **Training curves:** loss jumps or becomes NaN; learning rate that worked on a smaller model fails at scale.
- **Gradient norms:** global or per-layer \(\|\nabla\|\) tracked online (as in Pascanu et al.’s monitoring when clipping).
- **Optimizer state:** Adam’s \(v_t\) and effective step sizes can look normal right until a spike (see [adam_collapse](adam_collapse.md) for Adam-specific instability).

## Literature: detection and response

### Gradient clipping (general deep networks)

**Pascanu et al. (2013)** — *On the difficulty of training recurrent neural networks* — document exploding/vanishing gradients in RNNs and propose **norm-based clipping**: if \(\|g\| > \tau\), scale \(g \leftarrow \tau \cdot g / \|g\|\). They **detect** explosion by monitoring \(\|g\|\) and relate instability to the spectral radius of Jacobian products along time.

**Fix (math):** caps the linearized update \(\Delta\theta \propto g\) so a single bad minibatch cannot move parameters by an unbounded amount.

### Residual connections and normalization (depth)

**He et al. (2016)** — *Deep Residual Learning* — address optimization in very deep stacks by learning **residual** mappings \(F(x)+x\). Gradients flow through the identity skip; this **reduces** growth of gradient magnitude with depth compared to plain stacks.

**Fix (architecture):** reparameterize layers so the default signal path is additive, improving conditioning of backward propagation.

### Transformers and warmup

**Xiong et al. (2020)** — *On Layer Normalization in the Transformer Architecture* — show **Post-LN** transformers at initialization have **large gradients near the output**; a full learning rate without **warmup** is unstable. They **spot** this via mean-field analysis of gradient scale vs depth and empirically via failed runs without warmup.

**Fix (schedule + norm placement):** linear LR warmup; or **Pre-LN** (\(\mathrm{LN}\) inside the residual) for more stable initial gradients.

## References

- Pascanu, Mikolov, Bengio (2013). [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063).
- He, Zhang, Ren, Sun (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Xiong et al. (2020). [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html).
