"""Offline W&B run simulating Llama-scale per-checkpoint ``update_rank`` logging."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import wandb

LLAMA70B_RUN_ID = "bevllama70b1"
LLAMA70B_RUN_NAME = "llama70b-update-rank-scale"

# Meta Llama-2/3 70B-class defaults (GQA).
LLAMA70B_NUM_LAYERS = 80
LLAMA70B_NUM_Q_HEADS = 64
LLAMA70B_NUM_KV_HEADS = 8

# Twenty scalar snapshots (e.g. SFT checkpoints / eval saves).
LLAMA70B_DEFAULT_CHECKPOINT_STEPS: tuple[int, ...] = tuple(step * 100 for step in range(20))

METRIC_PREFIX = "train"
UPDATE_RANK_FAMILY = "update_rank"


def expected_update_rank_key_count(
    num_layers: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    include_global: bool = True,
) -> int:
    """Return the number of distinct ``update_rank`` keys for the Llama layout.

    Args:
        num_layers: Decoder layer count.
        num_q_heads: Query (and ``o_proj``) head slices per layer.
        num_kv_heads: Key/value head slices per layer (GQA).
        include_global: Count ``embed_tokens`` and ``lm_head``.

    Returns:
        Expected catalog size for ``update_rank`` only.
    """
    attn_per_layer = (2 * num_q_heads) + (2 * num_kv_heads)
    ffn_per_layer = 3
    global_keys = 2 if include_global else 0
    return global_keys + num_layers * (attn_per_layer + ffn_per_layer)


def iter_llama_update_rank_keys(
    num_layers: int = LLAMA70B_NUM_LAYERS,
    num_q_heads: int = LLAMA70B_NUM_Q_HEADS,
    num_kv_heads: int = LLAMA70B_NUM_KV_HEADS,
    *,
    prefix: str = METRIC_PREFIX,
) -> list[str]:
    """Build ``train/update_rank/...`` keys for a Llama-style causal LM.

    Head-sliced attention ranks use ``/head.N`` on Q/O and ``/kv_head.N`` on K/V.
    FFN and global matrices use one scalar per weight tensor.

    Args:
        num_layers: ``model.layers.0`` … ``model.layers.{N-1}``.
        num_q_heads: Slices for ``q_proj`` and ``o_proj``.
        num_kv_heads: Slices for ``k_proj`` and ``v_proj``.
        prefix: Metric key prefix (typically ``train``).

    Returns:
        Sorted list of metric keys.
    """
    family = f"{prefix}/{UPDATE_RANK_FAMILY}"
    keys: list[str] = []

    def add(key: str) -> None:
        keys.append(key)

    add(f"{family}/model.embed_tokens.weight")

    for layer in range(num_layers):
        base = f"model.layers.{layer}.self_attn"
        for head in range(num_q_heads):
            add(f"{family}/{base}.q_proj.weight/head.{head}")
            add(f"{family}/{base}.o_proj.weight/head.{head}")
        for kv_head in range(num_kv_heads):
            add(f"{family}/{base}.k_proj.weight/kv_head.{kv_head}")
            add(f"{family}/{base}.v_proj.weight/kv_head.{kv_head}")
        mlp = f"model.layers.{layer}.mlp"
        add(f"{family}/{mlp}.gate_proj.weight")
        add(f"{family}/{mlp}.up_proj.weight")
        add(f"{family}/{mlp}.down_proj.weight")

    add(f"{family}/model.lm_head.weight")
    return keys


def iter_llama_update_rank_keys_lazy(
    num_layers: int = LLAMA70B_NUM_LAYERS,
    num_q_heads: int = LLAMA70B_NUM_Q_HEADS,
    num_kv_heads: int = LLAMA70B_NUM_KV_HEADS,
    *,
    prefix: str = METRIC_PREFIX,
) -> Iterator[str]:
    """Yield keys without building the full list (for memory-conscious tools)."""
    yield from iter_llama_update_rank_keys(
        num_layers,
        num_q_heads,
        num_kv_heads,
        prefix=prefix,
    )


def metric_values_for_step(keys: list[str], step: int) -> dict[str, float]:
    """Deterministic scalar values for fixture runs (no randomness in CI)."""
    scale = 1.0 + step * 0.001
    out: dict[str, float] = {}
    for index, key in enumerate(keys):
        out[key] = ((index * 13 + step * 7) % 10_000) / 10_000.0 * scale
    return out


def llama70b_fixture_summary(
    num_layers: int = LLAMA70B_NUM_LAYERS,
    num_q_heads: int = LLAMA70B_NUM_Q_HEADS,
    num_kv_heads: int = LLAMA70B_NUM_KV_HEADS,
    *,
    num_checkpoints: int = len(LLAMA70B_DEFAULT_CHECKPOINT_STEPS),
) -> dict[str, int]:
    """Return expected counts for docs, tests, and bootstrap scripts.

    Args:
        num_layers: Decoder depth.
        num_q_heads: Query head slices.
        num_kv_heads: KV head slices.
        num_checkpoints: Number of ``wandb.log`` steps in the sim run.

    Returns:
        Summary dict with key and scalar-event totals.
    """
    keys_per_checkpoint = expected_update_rank_key_count(num_layers, num_q_heads, num_kv_heads)
    return {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "update_rank_keys": keys_per_checkpoint,
        "checkpoints": num_checkpoints,
        "scalar_events": keys_per_checkpoint * num_checkpoints,
        "attn_slices_per_layer": (2 * num_q_heads) + (2 * num_kv_heads),
        "ffn_keys_per_layer": 3,
    }


def bootstrap_llama70b_scale_scan_root(
    base: Path,
    *,
    num_layers: int = LLAMA70B_NUM_LAYERS,
    num_q_heads: int = LLAMA70B_NUM_Q_HEADS,
    num_kv_heads: int = LLAMA70B_NUM_KV_HEADS,
    steps: tuple[int, ...] = LLAMA70B_DEFAULT_CHECKPOINT_STEPS,
) -> Path:
    """Write an offline W&B run with Llama 70B-scale ``update_rank`` catalog.

    Args:
        base: Writable directory; created if missing.
        num_layers: Decoder depth (default 80).
        num_q_heads: Q/O head slices (default 64).
        num_kv_heads: K/V head slices (default 8).
        steps: Checkpoint steps to log (default 20 steps).

    Returns:
        Scan root path passed to ``bev serve --root``.
    """
    base.mkdir(parents=True, exist_ok=True)
    keys = iter_llama_update_rank_keys(num_layers, num_q_heads, num_kv_heads)
    summary = llama70b_fixture_summary(
        num_layers,
        num_q_heads,
        num_kv_heads,
        num_checkpoints=len(steps),
    )
    if len(keys) != summary["update_rank_keys"]:
        msg = f"key count mismatch: {len(keys)} != {summary['update_rank_keys']}"
        raise RuntimeError(msg)

    wandb_dir = base / "wandb"
    wandb.init(
        mode="offline",
        dir=str(wandb_dir),
        project="bev_llama70b_scale",
        name=LLAMA70B_RUN_NAME,
        id=LLAMA70B_RUN_ID,
        reinit=True,
    )
    for step in steps:
        wandb.log(metric_values_for_step(keys, step), step=step)
    wandb.finish()
    return base
