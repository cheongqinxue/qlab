# Attention sink (`attention_sink`)

## In this tutorial

For BERT MLM we **bias position and word embeddings** so token 0 attracts disproportionate attention (`apply_attention_sink_init` in `encoder_setup.py`). Signatures: high `train/attention_max_weight/*`, high `train/attention_head_imbalance/*`, often low attention entropy on affected layers.

## What goes wrong

A few positions (often **first token**, padding, or low-information patches) absorb a large fraction of attention mass. Other tokens receive little gradient through attention; representations become **over-dependent** on “sink” tokens. In ViTs these coincide with **high-norm outlier tokens** in background regions.

## How practitioners spot it

- **Attention maps:** one or few keys/columns dominate (visual heatmaps in ViT; [CLS] or BOS in language models).
- **Token norms:** outlier tokens with \(\|h_i\|\) far above the median (Darcet et al.).
- **Head imbalance:** ratio of max to min per-head peak attention (our `attention_head_imbalance` metric).
- **Downstream:** noisy feature maps, poor dense prediction, object-discovery failures (LOST-style methods) before mitigation.

## Literature: detection and response

### Vision Transformers — registers

**Darcet et al. (2023)** — *Vision Transformers Need Registers* — **detect** high-norm tokens in low-information areas; attention “artifacts” used as internal computation slots. **Fix:** append learnable **register tokens** after patch embedding (like extra [CLS] slots) so outliers move off image patches; discard registers at output.

**Math / mechanism:** extra sequence positions provide a dedicated subspace for global aggregation instead of hijacking arbitrary patches.

### Test-time registers (no retrain)

**Chen et al. (2025)** — *Vision Transformers Don't Need Trained Registers* — identify **register neurons** that concentrate high norms; at test time shift activations into an **untrained token** to mimic trained registers.

### LLMs — sink tokens at sequence start

**Xiao et al. (2023)** — *Efficient Streaming Language Models with Attention Sinks* — observe early tokens (e.g. BOS) receive persistent high attention in streaming LLMs; **retain** initial KV cache entries rather than sliding window on all tokens.

**Detection:** attention statistics on first positions across layers; ablations removing sink tokens hurt perplexity.

**Fix (inference):** treat sinks as structural anchors, not bugs—different from ViT artifact mitigation but same **peaked attention** phenomenon.

### LVLMs

Recent work (e.g. Kang et al., 2025; unified sink frameworks) **categorize** vision-emerged vs LLM-emerged sinks and **gate** their contribution rather than only masking them.

## References

- Darcet et al. (2023). [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).
- Chen et al. (2025). [Vision Transformers Don't Need Trained Registers](https://www.proceedings.com/content/085/085713-1891open.pdf).
- Xiao et al. (2023). [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453).
