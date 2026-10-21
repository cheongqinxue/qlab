# Weight decay on LayerNorm and bias (`decay_norm_bias`)

**Encoder MLM only** in this tutorial.

## In this tutorial

We apply **weight decay to LayerNorm parameters and biases** (`decay_norm_and_bias=True`), with elevated global decay (`0.15`). Standard BERT practice **excludes** them from L2 penalty.

## What goes wrong

LayerNorm scale \(\gamma\) and shift \(\beta\) are meant to **renormalize** activations; decaying \(\gamma,\beta\) fights the optimizer’s need to set scale/shift. Decaying **biases** in attention and MLP can shift baseline activations. Training becomes slower, less stable, or slightly worse MLM—subtle compared to `explode`, but a real **recipe bug**.

## How practitioners spot it

- **Compare to standard AdamW grouping:** inspect `optimizer.param_groups`—which tensors get `weight_decay > 0`.
- **Replication failures:** BERT/Transformer reproductions that decay all parameters often underperform official checkpoints.
- **No single W&B metric:** diagnosis is usually “we violated the known recipe,” not a dramatic spike.

## Literature: detection and response

### BERT pretraining recipe

**Devlin et al. (2019)** — BERT uses **AdamW** with weight decay on **weights** but **not** on LayerNorm or bias terms (as implemented in HuggingFace `Trainer` via `get_decay_parameter_names`).

**Fix:** partition parameters into decay vs no-decay groups; apply \(\lambda \|W\|^2\) only on appropriate weights.

### AdamW decoupled decay

**Loshchilov & Hutter (2019)** — decouple weight decay from gradient update; decay applies to parameters that benefit from L2 (weights), not normalization affine terms.

**Math:** \(\theta \leftarrow \theta - \eta(\nabla + \lambda \theta)\) only where \(\theta\) is a “weight” matrix, not \(\gamma,\beta\) of LN.

### Why LN is excluded (intuition)

LN already constrains activation scale: \(\hat x = (x-\mu)/\sigma\), \(y = \gamma \hat x + \beta\). Penalizing \(\|\gamma\|\) pushes toward \(\gamma \to 0\) (dead scaling) unless gradients oppose strongly—poor conditioning.

## References

- Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805).
- Loshchilov, Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
