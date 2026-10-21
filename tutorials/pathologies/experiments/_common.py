"""Shared model ids, datasets, pathology presets, and Trainer helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from transformers import TrainingArguments, set_seed as hf_set_seed

ENCODER_MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
ENCODER_MODEL_DEEP_ID = "google/bert_uncased_L-12_H-768_A-12"
LLM_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
WIKITEXT_PATH = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
AG_NEWS_PATH = "fancyzhx/ag_news"
SMOLTALK_PATH = "HuggingFaceTB/smoltalk"
GSM8K_PATH = "openai/gsm8k"
GSM8K_CONFIG = "main"

ABLATION_VARIANTS = frozenset({"standard", "no_scale", "no_norm", "no_residual"})


class PathologyCategory(str, Enum):
    """Whether a mode is a training pathology or a best-practice violation."""

    TRAINING = "training"
    BEST_PRACTICE = "best_practice"


BASELINE_MODE = "baseline"

TRAINING_PATHOLOGIES = frozenset(
    {
        "explode",
        "vanish",
        "adam_collapse",
        "attention_sink",
        "rank_collapse",
        "logit_collapse",
        "layer_grad_imbalance",
        "representation_collapse",
    },
)

BEST_PRACTICE_VIOLATIONS = frozenset(
    {
        "no_warmup",
        "decay_norm_bias",
        "bad_mlm_mask",
    },
)

PATHOLOGY_MODES = TRAINING_PATHOLOGIES | BEST_PRACTICE_VIOLATIONS
FAILURE_MODES = PATHOLOGY_MODES | {BASELINE_MODE}


def set_run_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and HuggingFace RNG state before a training run.

    Args:
        seed: Integer seed shared by weight init, dataloaders, and synthetic data.
    """
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)


def pathology_category(mode: str) -> PathologyCategory | None:
    """Return the taxonomy bucket for a failure mode, or ``None`` for baseline."""
    if mode == BASELINE_MODE:
        return None
    if mode in TRAINING_PATHOLOGIES:
        return PathologyCategory.TRAINING
    if mode in BEST_PRACTICE_VIOLATIONS:
        return PathologyCategory.BEST_PRACTICE
    msg = f"Unknown failure mode {mode!r}"
    raise KeyError(msg)


def failure_modes_for_family(family: str) -> frozenset[str]:
    """Return failure modes supported by a profile-table family."""
    modes = {BASELINE_MODE, "explode", "vanish", "adam_collapse", "logit_collapse"}
    if family in ("encoder", "ablation"):
        modes |= {
            "attention_sink",
            "rank_collapse",
            "layer_grad_imbalance",
            "representation_collapse",
        }
    if family == "encoder":
        modes |= BEST_PRACTICE_VIOLATIONS
    return frozenset(modes)


@dataclass(frozen=True)
class FailureProfile:
    """Hyperparameters and hooks chosen to produce a visible training regime."""

    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    warmup_ratio: float
    label: str
    category: PathologyCategory | None = None
    adam_epsilon: float = 1e-8
    min_steps: int = 100
    mlm_probability: float | None = None
    decay_norm_and_bias: bool = False
    embedding_lr_multiplier: float = 1.0
    encoder_lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.0
    attention_sink_init: bool = False
    trainable_prefixes: tuple[str, ...] = ()


BASELINE = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label="baseline",
    min_steps=100,
)

EXPLODE = FailureProfile(
    learning_rate=5e-3,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="explode",
    category=PathologyCategory.TRAINING,
    min_steps=100,
)
VANISH = FailureProfile(
    learning_rate=1e-7,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.0,
    label="vanish",
    category=PathologyCategory.TRAINING,
    min_steps=150,
)
ADAM_COLLAPSE = FailureProfile(
    learning_rate=1e-3,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="adam_collapse",
    category=PathologyCategory.TRAINING,
    adam_epsilon=1e-16,
    min_steps=100,
)

ATTENTION_SINK = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label="attention_sink",
    category=PathologyCategory.TRAINING,
    min_steps=200,
    attention_sink_init=True,
)
RANK_COLLAPSE = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.35,
    warmup_ratio=0.1,
    label="rank_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)
LOGIT_COLLAPSE = FailureProfile(
    learning_rate=8e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="logit_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=250,
)
LAYER_GRAD_IMBALANCE = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label="layer_grad_imbalance",
    category=PathologyCategory.TRAINING,
    min_steps=150,
    embedding_lr_multiplier=40.0,
    encoder_lr_multiplier=0.02,
    head_lr_multiplier=40.0,
)
REPRESENTATION_COLLAPSE = FailureProfile(
    learning_rate=5e-3,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="representation_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
    trainable_prefixes=("bert.embeddings.word_embeddings",),
)

NO_WARMUP = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.0,
    label="no_warmup",
    category=PathologyCategory.BEST_PRACTICE,
    min_steps=150,
)
DECAY_NORM_BIAS = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.15,
    warmup_ratio=0.1,
    label="decay_norm_bias",
    category=PathologyCategory.BEST_PRACTICE,
    min_steps=250,
    decay_norm_and_bias=True,
)
BAD_MLM_MASK = FailureProfile(
    learning_rate=5e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label="bad_mlm_mask",
    category=PathologyCategory.BEST_PRACTICE,
    min_steps=200,
    mlm_probability=0.02,
)

BILSTM_EXPLODE = FailureProfile(
    learning_rate=1e-3,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="explode",
    category=PathologyCategory.TRAINING,
)
BILSTM_VANISH = FailureProfile(
    learning_rate=1e-3,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="vanish",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)
BILSTM_ADAM_COLLAPSE = FailureProfile(
    learning_rate=5e-3,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="adam_collapse",
    category=PathologyCategory.TRAINING,
    adam_epsilon=1e-16,
)
BILSTM_LOGIT_COLLAPSE = FailureProfile(
    learning_rate=5e-3,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="logit_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)

ABLATION_VANISH = FailureProfile(
    learning_rate=1e-10,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="vanish",
    category=PathologyCategory.TRAINING,
    min_steps=150,
)
ABLATION_ATTENTION_SINK = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="attention_sink",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)
ABLATION_RANK_COLLAPSE = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1.0,
    weight_decay=0.4,
    warmup_ratio=0.0,
    label="rank_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=350,
)
ABLATION_LOGIT_COLLAPSE = FailureProfile(
    learning_rate=2e-2,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="logit_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)
ABLATION_LAYER_GRAD_IMBALANCE = FailureProfile(
    learning_rate=1e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="layer_grad_imbalance",
    category=PathologyCategory.TRAINING,
    min_steps=150,
    embedding_lr_multiplier=50.0,
    encoder_lr_multiplier=0.01,
    head_lr_multiplier=50.0,
)
ABLATION_REPRESENTATION_COLLAPSE = FailureProfile(
    learning_rate=1e-2,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="representation_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
    trainable_prefixes=("input_proj",),
)

LLM_SFT_BASELINE = FailureProfile(
    learning_rate=2e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_ratio=0.03,
    label="baseline",
    min_steps=100,
)
LLM_SFT_EXPLODE = FailureProfile(
    learning_rate=2e-3,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="explode",
    category=PathologyCategory.TRAINING,
)
LLM_SFT_VANISH = FailureProfile(
    learning_rate=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.0,
    label="vanish",
    category=PathologyCategory.TRAINING,
    min_steps=150,
)
LLM_SFT_ADAM_COLLAPSE = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="adam_collapse",
    category=PathologyCategory.TRAINING,
    adam_epsilon=1e-16,
)
LLM_SFT_LOGIT_COLLAPSE = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="logit_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)

LLM_GRPO_BASELINE = FailureProfile(
    learning_rate=5e-6,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="baseline",
    min_steps=100,
)
LLM_GRPO_EXPLODE = FailureProfile(
    learning_rate=5e-4,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="explode",
    category=PathologyCategory.TRAINING,
)
LLM_GRPO_VANISH = FailureProfile(
    learning_rate=1e-9,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="vanish",
    category=PathologyCategory.TRAINING,
    min_steps=150,
)
LLM_GRPO_ADAM_COLLAPSE = FailureProfile(
    learning_rate=1e-4,
    max_grad_norm=1e9,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="adam_collapse",
    category=PathologyCategory.TRAINING,
    adam_epsilon=1e-16,
)
LLM_GRPO_LOGIT_COLLAPSE = FailureProfile(
    learning_rate=2e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    warmup_ratio=0.0,
    label="logit_collapse",
    category=PathologyCategory.TRAINING,
    min_steps=200,
)

_ENCODER_PATHOLOGIES = {
    "attention_sink": ATTENTION_SINK,
    "rank_collapse": RANK_COLLAPSE,
    "logit_collapse": LOGIT_COLLAPSE,
    "layer_grad_imbalance": LAYER_GRAD_IMBALANCE,
    "representation_collapse": REPRESENTATION_COLLAPSE,
    "no_warmup": NO_WARMUP,
    "decay_norm_bias": DECAY_NORM_BIAS,
    "bad_mlm_mask": BAD_MLM_MASK,
}

_PROFILE_TABLE: dict[str, dict[str, FailureProfile]] = {
    "encoder": {
        "baseline": BASELINE,
        "explode": EXPLODE,
        "vanish": VANISH,
        "adam_collapse": ADAM_COLLAPSE,
        **_ENCODER_PATHOLOGIES,
    },
    "bilstm": {
        "baseline": BASELINE,
        "explode": BILSTM_EXPLODE,
        "vanish": BILSTM_VANISH,
        "adam_collapse": BILSTM_ADAM_COLLAPSE,
        "logit_collapse": BILSTM_LOGIT_COLLAPSE,
    },
    "llm_sft": {
        "baseline": LLM_SFT_BASELINE,
        "explode": LLM_SFT_EXPLODE,
        "vanish": LLM_SFT_VANISH,
        "adam_collapse": LLM_SFT_ADAM_COLLAPSE,
        "logit_collapse": LLM_SFT_LOGIT_COLLAPSE,
    },
    "llm_grpo": {
        "baseline": LLM_GRPO_BASELINE,
        "explode": LLM_GRPO_EXPLODE,
        "vanish": LLM_GRPO_VANISH,
        "adam_collapse": LLM_GRPO_ADAM_COLLAPSE,
        "logit_collapse": LLM_GRPO_LOGIT_COLLAPSE,
    },
    "ablation": {
        "baseline": BASELINE,
        "explode": EXPLODE,
        "vanish": ABLATION_VANISH,
        "adam_collapse": ADAM_COLLAPSE,
        "attention_sink": ABLATION_ATTENTION_SINK,
        "rank_collapse": ABLATION_RANK_COLLAPSE,
        "logit_collapse": ABLATION_LOGIT_COLLAPSE,
        "layer_grad_imbalance": ABLATION_LAYER_GRAD_IMBALANCE,
        "representation_collapse": ABLATION_REPRESENTATION_COLLAPSE,
    },
}


def effective_max_steps(config_max_steps: int, profile: FailureProfile) -> int:
    """Use the longer of CLI steps and the profile's recommended minimum."""
    return max(config_max_steps, profile.min_steps)


def resolve_failure_profile(family: str, failure_mode: str) -> FailureProfile:
    """Look up the hyperparameter preset for an experiment family and failure mode.

    Args:
        family: Profile table key (``encoder``, ``bilstm``, ``llm_sft``, etc.).
        failure_mode: One of :data:`FAILURE_MODES`.

    Returns:
        Matching :class:`FailureProfile`.

    Raises:
        KeyError: If the family or failure mode is unknown.
    """
    if failure_mode not in FAILURE_MODES:
        msg = f"Unknown failure_mode {failure_mode!r}. Choose from: {sorted(FAILURE_MODES)}"
        raise KeyError(msg)
    table = _PROFILE_TABLE.get(family)
    if table is None or failure_mode not in table:
        msg = f"No failure profile for family={family!r}, mode={failure_mode!r}"
        raise KeyError(msg)
    return table[failure_mode]


def training_arguments(
    *,
    output_dir: str,
    profile: FailureProfile,
    max_steps: int,
    seed: int,
    run_name: str | None = None,
    per_device_train_batch_size: int = 8,
    logging_steps: int = 1,
    warmup_ratio: float | None = None,
    label_names: list[str] | None = None,
) -> TrainingArguments:
    """Build :class:`~transformers.TrainingArguments` with W&B logging."""
    ratio = profile.warmup_ratio if warmup_ratio is None else warmup_ratio
    warmup_steps = int(max_steps * ratio)
    return TrainingArguments(
        output_dir=output_dir,
        label_names=label_names,
        run_name=run_name or profile.label,
        report_to=["wandb"],
        logging_steps=logging_steps,
        save_strategy="no",
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=profile.learning_rate,
        max_grad_norm=profile.max_grad_norm,
        weight_decay=profile.weight_decay,
        warmup_steps=warmup_steps,
        adam_epsilon=profile.adam_epsilon,
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
