# Training pathologies — reference notes

One page per **failure mode** in `experiments/_common.py`. Each file links the tutorial’s W&B signatures to how the wider ML literature **detects** and **mitigates** similar failure modes. This is not exhaustive across every architecture; citations are representative.

| Mode | Category | Doc |
|------|----------|-----|
| `explode` | Training | [explode.md](explode.md) |
| `vanish` | Training | [vanish.md](vanish.md) |
| `adam_collapse` | Training | [adam_collapse.md](adam_collapse.md) |
| `attention_sink` | Training | [attention_sink.md](attention_sink.md) |
| `rank_collapse` | Training | [rank_collapse.md](rank_collapse.md) |
| `logit_collapse` | Training | [logit_collapse.md](logit_collapse.md) |
| `layer_grad_imbalance` | Training | [layer_grad_imbalance.md](layer_grad_imbalance.md) |
| `representation_collapse` | Training | [representation_collapse.md](representation_collapse.md) |
| `no_warmup` | Best practice (encoder MLM) | [no_warmup.md](no_warmup.md) |
| `decay_norm_bias` | Best practice (encoder MLM) | [decay_norm_bias.md](decay_norm_bias.md) |
| `bad_mlm_mask` | Best practice (encoder MLM) | [bad_mlm_mask.md](bad_mlm_mask.md) |

**Baseline** runs use sane hyperparameters and are the comparison point for every signature in the main [README](../../README.md).
