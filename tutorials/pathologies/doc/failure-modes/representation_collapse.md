# Representation collapse (`representation_collapse`)

## In this tutorial

We **freeze** almost all parameters and train only a small prefix (e.g. **word embeddings only** on BERT, or **input_proj** on the ablation encoder). Watch rising **embedding anisotropy** (`train/embedding_anisotropy`), poor use of depth, and weak signals in encoder layers (near-zero grads).

## What goes wrong

**Collapse** means representations lose diversity: embeddings or features occupy a **low-dimensional cone** or constant direction; cosine similarity between unrelated tokens becomes high. Contrastive learning without negatives collapses to a **constant** embedding (Chen & He, SimCLR analysis). Our mode mimics **bottleneck fine-tuning** where the body cannot adapt.

## How practitioners spot it

- **Embedding anisotropy / singular values** of the embedding matrix (our metric; related to “anisotropy” literature in word2vec and transformers).
- **Cosine similarity histogram** between random token pairs—mass near 1.0.
- **Contrastive SSL:** trivial solution \(f(x)=\mathrm{const}\) (detected when loss goes to optimum without learning).
- **Neural collapse:** features collapse to class means (supervised, late training)—different trigger, similar “lack of spread.”

## Literature: detection and response

### Contrastive learning collapse

**Chen et al. (2020)** — *A Simple Framework for Contrastive Learning (SimCLR)* — **negative pairs** and large batches **prevent** trivial constant representations. **Detection:** train loss near \(-\infty\) with collapsed embeddings if negatives removed.

**Fix (loss):** InfoNCE-style repulsion between different examples.

### Dimensional / rank collapse in SSL

**Jing et al. (2022)** — embeddings span subspace of dimension \(< d\); **detect** via covariance spectrum.

### Neural collapse (supervised)

**Papyan et al. (2020)** — features collapse to **class means** forming ETF; **detect** via geometry of penultimate layer. **Harun et al. (2025)** control collapse strength for OOD vs transfer.

### Mitigation when fine-tuning little data

**ULMFiT**, **gradual unfreezing**, **weight decay** on embeddings, **LayerNorm tuning** only (BitFit, LoRA) — avoid training **only** embeddings while keeping a mismatched body (our pathology).

## References

- Chen et al. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).
- Jing et al. (2022). [Understanding Dimensional Collapse in Contrastive SSL](https://arxiv.org/abs/2110.09348).
- Papyan et al. (2020). [Prevalence of neural collapse](https://arxiv.org/abs/2006.07739).
