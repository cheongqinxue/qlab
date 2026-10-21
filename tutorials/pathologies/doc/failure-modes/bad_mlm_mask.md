# Bad MLM masking rate (`bad_mlm_mask`)

**Encoder MLM only** in this tutorial.

## In this tutorial

We set **`mlm_probability=0.02`** instead of BERT’s **15%**. The model sees too few masked tokens per sequence; MLM loss has less signal per step and pretraining-style objectives behave poorly.

## What goes wrong

Masked language modeling trains the model to predict tokens hidden by `[MASK]` (and sometimes other corruption). Too **few** masks → weak denoising signal, underuse of context; too **many** masks → optimization difficulty and train/inference mismatch. BERT’s 15% is a tuned compromise.

## How practitioners spot it

- **Slower MLM loss decrease** vs standard masking on same data/steps.
- **Downstream gap** when pretraining with wrong mask rate (reported in WWM and RoBERTa ablations).
- **Our mode:** compare `bad_mlm_mask` to `baseline` on `encoder_mlm` with identical steps.

## Literature: detection and response

### BERT masking protocol

**Devlin et al. (2019)** — mask **15%** of WordPiece tokens; of those, **80%** `[MASK]`, **10%** random word, **10%** unchanged. **Detection** in ablations: dev perplexity and GLUE after pretraining.

**Fix:** restore 15% + 80/10/10 rule.

### Whole Word Masking

**Cui et al. (2019)**; **Joshi et al. (2020)** — mask **entire words** (all WordPieces) instead of random subwords; improves Chinese/English pretraining. **Spot** improvement on NER and QA; masking **unit** matters, not only rate.

### RoBERTa

**Liu et al. (2019)** — remove NSP, train longer, **dynamic masking** each epoch; still ~15% mask probability. Shows **recipe** sensitivity beyond architecture.

**Fix (data):** dynamic masks; correct rate; optional whole-word masking.

## References

- Devlin et al. (2019). [BERT](https://arxiv.org/abs/1810.04805).
- Liu et al. (2019). [RoBERTa](https://arxiv.org/abs/1907.11692).
- Joshi et al. (2020). [SpanBERT / whole word masking](https://arxiv.org/abs/1907.10529).
