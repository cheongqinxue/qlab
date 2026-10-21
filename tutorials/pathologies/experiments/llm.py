"""SmolLM2 SFT and GRPO — one experiment each, all failure modes via CLI."""

from __future__ import annotations

from functools import partial

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, SFTConfig

from pathologies.experiment import Experiment, RunConfig, register_experiment
from pathologies.experiments._common import (
    GSM8K_CONFIG,
    GSM8K_PATH,
    LLM_MODEL_ID,
    SMOLTALK_PATH,
    effective_max_steps,
    failure_modes_for_family,
    resolve_failure_profile,
)
from pathologies.experiments.pathology_trainer import PathologyGRPOTrainer, PathologySFTTrainer
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import build_trainer_callbacks


def _load_smoltalk(max_samples: int):
    dataset = load_dataset(SMOLTALK_PATH, "all", split="train")
    return dataset.select(range(min(len(dataset), max_samples)))


def _format_smoltalk(example: dict, tokenizer) -> dict:
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def _load_llm_for_telemetry() -> AutoModelForCausalLM:
    """Load SmolLM with eager attention so forward hooks can read attentions."""
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        attn_implementation="eager",
    )
    model.config.output_attentions = True
    model.config.use_cache = False
    return model


def run_llm_sft(config: RunConfig) -> None:
    """SFT SmolLM2; failure mode from ``config.failure_mode``."""
    profile = resolve_failure_profile("llm_sft", config.failure_mode)
    max_steps = effective_max_steps(config.max_steps, profile)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_llm_for_telemetry()
    telemetry = config.telemetry or TelemetryConfig.llm()
    callbacks, _collector = build_trainer_callbacks(model, telemetry)
    dataset = _load_smoltalk(config.max_samples)
    dataset = dataset.map(
        partial(_format_smoltalk, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
    )

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        run_name=config.run_name or profile.label,
        report_to=["wandb"],
        logging_steps=1,
        save_strategy="no",
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=profile.learning_rate,
        max_grad_norm=profile.max_grad_norm,
        weight_decay=profile.weight_decay,
        warmup_steps=int(max_steps * profile.warmup_ratio),
        adam_epsilon=profile.adam_epsilon,
        seed=config.seed,
        data_seed=config.seed,
        fp16=False,
        bf16=False,
        dataset_text_field="text",
        max_length=256,
        packing=False,
        dataloader_num_workers=0,
    )

    trainer = PathologySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()


def _length_reward(completions: list[str], **kwargs) -> list[float]:
    del kwargs
    return [min(len(completion) / 128.0, 1.0) for completion in completions]


def _gsm8k_to_prompt(example: dict) -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Solve the following grade-school math problem.\n\n"
                    f"{example['question']}"
                ),
            }
        ],
    }


def run_llm_grpo(config: RunConfig) -> None:
    """GRPO on SmolLM2; failure mode from ``config.failure_mode``."""
    profile = resolve_failure_profile("llm_grpo", config.failure_mode)
    max_steps = min(effective_max_steps(config.max_steps, profile), 30)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_llm_for_telemetry()
    telemetry = config.telemetry or TelemetryConfig.llm()
    callbacks, _collector = build_trainer_callbacks(model, telemetry)
    dataset = load_dataset(GSM8K_PATH, GSM8K_CONFIG, split="train")
    dataset = dataset.select(range(min(len(dataset), min(config.max_samples, 128))))
    dataset = dataset.map(_gsm8k_to_prompt, remove_columns=dataset.column_names)

    training_args = GRPOConfig(
        output_dir=str(config.output_dir),
        run_name=config.run_name or profile.label,
        report_to=["wandb"],
        logging_steps=1,
        save_strategy="no",
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=profile.learning_rate,
        max_grad_norm=profile.max_grad_norm,
        weight_decay=profile.weight_decay,
        warmup_steps=int(max_steps * profile.warmup_ratio),
        adam_epsilon=profile.adam_epsilon,
        seed=config.seed,
        data_seed=config.seed,
        num_generations=2,
        max_completion_length=64,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
    )

    trainer = PathologyGRPOTrainer(
        model=model,
        reward_funcs=_length_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()


register_experiment(
    Experiment(
        name="llm_sft",
        description="SmolLM2-135M-Instruct SFT on SmolTalk; --failure-mode selects pathology.",
        model_family="llm",
        training_kind="sft",
        profile_family="llm_sft",
        failure_modes=failure_modes_for_family("llm_sft"),
        default_telemetry=TelemetryConfig.llm,
        run=run_llm_sft,
    )
)
register_experiment(
    Experiment(
        name="llm_grpo",
        description="SmolLM2 GRPO on GSM8K; --failure-mode selects pathology.",
        model_family="llm",
        training_kind="grpo",
        profile_family="llm_grpo",
        failure_modes=failure_modes_for_family("llm_grpo"),
        default_telemetry=TelemetryConfig.llm,
        run=run_llm_grpo,
    )
)
