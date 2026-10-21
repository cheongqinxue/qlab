# Gradient vanishing (`vanish`)

## In this tutorial

We use an **extremely small learning rate**, sometimes **high weight decay**, and **no warmup**. Loss barely moves; `train/update_to_weight_ratio` stays tiny; per-layer `train/grad_norm/layer.*` can show depth-wise attenuation (especially on BiLSTM `vanish`, which uses a deep stack and long sequences).

## What goes wrong

Gradients shrink as they propagate backward (product of Jacobians \(< 1\)), so early layers or distant time steps receive negligible updates. The model looks “stuck” even though loss is finite.

## How practitioners spot it

- **Flat training loss** with non-zero but tiny parameter updates.
- **Per-layer gradient norms** decreasing from output to input (depth) or from last to first LSTM layer.
- **Activation scale:** very small activations in early layers in RNNs (Bengio et al., Pascanu et al.).

## Literature: detection and response

### RNNs and gating

**Hochreiter & Schmidhuber (1997)** — *Long Short-Term Memory* — introduce **constant-error carousel** and gating so gradients can flow across many time steps without repeated squashing. **Detection** in their setting: standard RNNs fail on long-horizon tasks; LSTM succeeds.

**Fix (architecture):** multiplicative gates control what enters the cell state; backprop through the cell avoids repeated tanh saturation on the main path.

**Bengio et al. (1994)**; **Pascanu et al. (2013)** — formalize vanishing/exploding in RNNs via Jacobian products; clipping addresses explosion; architecture (LSTM, GRU) addresses vanishing.

### Depth in feedforward and CNN stacks

**He et al. (2016)** — residual connections ease gradient flow to earlier blocks (same mechanism as for explosion).

### Transformers

**Xiong et al. (2020)** — Post-LN can yield **ill-conditioned** gradient scales at init (often the opposite problem—large near output); **Pre-LN** improves gradient behavior at initialization. Deeper Post-LN models are also discussed in later work (e.g. Peri-LN) as risking **weaker** gradient signals in very deep settings.

**Fix:** Pre-LN, careful scaling (e.g. **μP** / width-dependent LR in some large-model recipes), or auxiliary losses—not covered in our tiny BERT runs but common at scale.

## References

- Hochreiter, Schmidhuber (1997). [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
- Bengio, Simard, Frasconi (1994). [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181).
- Pascanu et al. (2013). [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063).
- Xiong et al. (2020). [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html).
