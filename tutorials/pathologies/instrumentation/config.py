"""Configuration for attachable training telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TelemetryConfig:
    """Select which training health signals to log to W&B / Trainer.

    Pass to :func:`build_trainer_callbacks` or :func:`attach_telemetry`.
    All fields default off except core gradient metrics — opt in to the rest.
    """

    log_grad_norm: bool = True
    log_update_ratio: bool = True
    log_activation_rms: bool = False
    activation_modules: list[str] = field(default_factory=list)
    log_attention: bool = False
    attention_modules: list[str] = field(default_factory=list)
    log_attention_head_imbalance: bool = False
    log_logits_entropy: bool = False
    logits_module: str | None = None
    log_effective_rank_on_checkpoint: bool = False
    log_effective_rank_each_step: bool = False
    effective_rank_parameters: list[str] = field(default_factory=list)
    log_update_rank_each_step: bool = False
    update_rank_parameters: list[str] = field(default_factory=list)
    log_adam_second_moment: bool = False
    log_adam_health: bool = False
    log_layer_grad_norms: bool = False
    log_depth_grad_norms: bool = False
    log_embedding_anisotropy: bool = False
    embedding_parameter: str | None = None

    @classmethod
    def none(cls) -> TelemetryConfig:
        """Disable all telemetry."""
        return cls(
            log_grad_norm=False,
            log_update_ratio=False,
        )

    @classmethod
    def minimal(cls) -> TelemetryConfig:
        """Global grad norm and update ratio only."""
        return cls()

    @classmethod
    def encoder_mlm(cls) -> TelemetryConfig:
        """Diagnostics suited to tiny BERT MLM runs."""
        rank_parameters = [
            "bert.encoder.layer.*.attention.self.query.weight",
            "bert.encoder.layer.*.attention.self.key.weight",
        ]
        return cls(
            log_activation_rms=True,
            activation_modules=[
                "bert.encoder.layer.?.output.LayerNorm",
                "bert.embeddings.LayerNorm",
            ],
            log_attention=True,
            attention_modules=["bert.encoder.layer.?.attention.self"],
            log_attention_head_imbalance=True,
            log_logits_entropy=True,
            logits_module="cls.predictions",
            log_effective_rank_on_checkpoint=True,
            log_effective_rank_each_step=True,
            effective_rank_parameters=rank_parameters,
            log_update_rank_each_step=True,
            update_rank_parameters=rank_parameters,
            log_adam_second_moment=True,
            log_adam_health=True,
            log_layer_grad_norms=True,
            log_depth_grad_norms=True,
            log_embedding_anisotropy=True,
            embedding_parameter="bert.embeddings.word_embeddings.weight",
        )

    @classmethod
    def ablation_transformer(cls) -> TelemetryConfig:
        """Diagnostics for custom ablation encoder blocks."""
        return cls(
            log_activation_rms=True,
            activation_modules=["blocks.[0-9]", "input_proj", "head"],
            log_attention=True,
            attention_modules=["blocks.[0-9].attn"],
            log_logits_entropy=True,
            logits_module="head",
        )

    @classmethod
    def llm(cls) -> TelemetryConfig:
        """Diagnostics for SmolLM / Llama-style causal LMs (SFT and GRPO)."""
        rank_parameters = [
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
        ]
        return cls(
            log_activation_rms=True,
            activation_modules=[
                "model.norm",
                "model.layers.0.post_attention_layernorm",
            ],
            log_attention=True,
            attention_modules=[
                "model.layers.0.self_attn",
                "model.layers.1.self_attn",
            ],
            log_attention_head_imbalance=True,
            log_logits_entropy=True,
            logits_module="re:(model\\.)?lm_head",
            log_effective_rank_on_checkpoint=True,
            log_effective_rank_each_step=True,
            effective_rank_parameters=rank_parameters,
            log_update_rank_each_step=True,
            update_rank_parameters=rank_parameters,
            log_adam_second_moment=True,
            log_adam_health=True,
            log_depth_grad_norms=True,
        )

    @classmethod
    def bilstm(cls) -> TelemetryConfig:
        """Diagnostics for the custom BiLSTM classifier."""
        return cls(
            log_activation_rms=True,
            activation_modules=["lstm", "classifier"],
            log_logits_entropy=True,
            logits_module="classifier",
            log_adam_second_moment=True,
            log_adam_health=True,
            log_layer_grad_norms=True,
            log_depth_grad_norms=True,
        )
