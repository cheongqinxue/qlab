# Layer-wise gradient imbalance (`layer_grad_imbalance`)

## In this tutorial

We use **different learning-rate multipliers** for embedding, encoder/body, and head (e.g. embedding and head ×40, encoder ×0.02). Signatures: `train/update_ratio/embedding` and `train/update_ratio/head` much larger than `train/update_ratio/encoder`; uneven `train/grad_norm/layer.*`.

## What goes wrong

Not all layers should move at the same speed. If **embeddings** or **heads** update too fast relative to **body** layers, the model fits superficial cues or output scaling while representation layers lag—poor transfer and unstable fine-tuning. If the body updates too fast with frozen or slow heads, features become incompatible with the classifier.

## How practitioners spot it

- **Per-group update norms** (our `update_ratio/*` and grouped grad norms).
- **Layer-wise LR sweeps:** only some layers improve val loss when LR is tuned per block.
- **Fine-tuning practice:** discriminative fine-tuning in ULMFiT (Howard & Ruder) — observe which layers need which LR.

## Literature: detection and response

### Discriminative fine-tuning (NLP)

**Howard & Ruder (2018)** — *Universal Language Model Fine-tuning* — **detect** that lower layers need **smaller** LRs than top layers during fine-tune. **Fix:** **discriminative slanted triangular LR**: layer \(l\) gets \(\eta \cdot \xi^{L-l}\) with \(\xi < 1\).

**Math:** same global step, but effective update \(\|\Delta W_l\| \propto \eta_l \|\nabla_l\|\); scaling \(\eta_l\) balances representation vs task head.

### Transformer pretraining (Post-LN vs Pre-LN)

**Xiong et al. (2020)** — at initialization, **gradient scale varies strongly by depth** in Post-LN (large near output). **Warmup** and **Pre-LN** are fixes for **imbalance at init**, not arbitrary bad multipliers like our pathology mode.

### Large model training

**Layer-wise adaptive methods** (e.g. LARS, custom block LR in some vision and LM recipes) and **embedding LR scaling** in fine-tuning (smaller LR on embeddings, larger on adapters) are common **fixes** once imbalance is **detected** via grad norm logging.

**Our tutorial mode** deliberately creates **bad** imbalance; literature focuses on **correcting** natural imbalance.

## References

- Howard, Ruder (2018). [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146).
- Xiong et al. (2020). [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html).
