"""Deterministic repeat-then-shift batch schedules for pathology runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_text_blocks(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
) -> list[dict[str, list[int]]]:
    """Split one document into fixed-length blocks without random cropping.

    Args:
        text: Raw document string.
        tokenizer: HuggingFace tokenizer.
        block_size: Sequence length per block.

    Returns:
        List of feature dicts with ``input_ids`` and ``attention_mask``.
    """
    tokenized = tokenizer(text, truncation=False)
    input_ids = tokenized["input_ids"]
    if len(input_ids) < block_size:
        return []
    trimmed = len(input_ids) // block_size * block_size
    blocks: list[dict[str, list[int]]] = []
    for start in range(0, trimmed, block_size):
        chunk = input_ids[start : start + block_size]
        blocks.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * block_size,
            }
        )
    return blocks


def _apply_fixed_mlm_mask(
    features: dict[str, list[int]],
    tokenizer: PreTrainedTokenizerBase,
    mlm_probability: float,
    seed: int,
) -> dict[str, Any]:
    """Add MLM ``labels`` with a fixed RNG seed (same masks every repeat step)."""
    input_ids = np.array(features["input_ids"], dtype=np.int64)
    labels = input_ids.copy()
    probability_matrix = np.full(labels.shape, mlm_probability)
    rng = np.random.default_rng(seed)
    masked_indices = rng.random(labels.shape) < probability_matrix
    special = tokenizer.all_special_ids
    masked_indices = masked_indices & ~np.isin(labels, special)
    labels[~masked_indices] = -100
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        msg = "Tokenizer has no mask_token_id; MLM requires a masked-LM tokenizer."
        raise ValueError(msg)
    input_ids[masked_indices] = mask_token_id
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": features["attention_mask"],
        "labels": labels.tolist(),
    }


def pick_repeat_shift_source_rows(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    seed: int,
) -> tuple[int, int]:
    """Choose two distinct WikiText rows that each tokenize to at least one block.

    Args:
        texts: Training strings (same order as the loaded dataset).
        tokenizer: HuggingFace tokenizer.
        block_size: Minimum token length per chosen row.
        seed: Deterministic tie-breaking when picking among viable rows.

    Returns:
        ``(repeat_row_index, shift_row_index)`` into ``texts``.
    """
    viable: list[int] = []
    for index, text in enumerate(texts):
        if not text.strip():
            continue
        length = len(tokenizer(text, truncation=False)["input_ids"])
        if length >= block_size:
            viable.append(index)
    if len(viable) < 2:
        msg = (
            f"Need at least two WikiText rows with >={block_size} tokens; "
            f"found {len(viable)}."
        )
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    rng.shuffle(viable)
    return int(viable[0]), int(viable[1])


def build_repeat_shift_batch_features(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    block_size: int,
    repeat_row: int,
    shift_row: int,
    mlm_probability: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one training batch for the repeat phase and one for the final shift step.

    Args:
        texts: Corpus aligned with ``repeat_row`` / ``shift_row`` indices.
        tokenizer: MLM tokenizer.
        batch_size: Batch dimension for both phases.
        block_size: Tokens per sequence.
        repeat_row: WikiText row for steps ``0 .. num_steps-2``.
        shift_row: WikiText row for the final step.
        mlm_probability: BERT-style mask probability.
        seed: Base seed for MLM masks (repeat masks use ``seed``; shift uses ``seed + 1``).

    Returns:
        Two lists of length ``batch_size`` with ``input_ids``, ``attention_mask``, ``labels``.
    """
    repeat_blocks = tokenize_text_blocks(texts[repeat_row], tokenizer, block_size)
    shift_blocks = tokenize_text_blocks(texts[shift_row], tokenizer, block_size)
    if len(repeat_blocks) < 1 or len(shift_blocks) < 1:
        msg = "Repeat and shift rows must each yield at least one block."
        raise ValueError(msg)

    def cycle_blocks(blocks: list[dict[str, list[int]]], count: int) -> list[dict[str, list[int]]]:
        return [blocks[index % len(blocks)] for index in range(count)]

    repeat_raw = cycle_blocks(repeat_blocks, batch_size)
    shift_raw = cycle_blocks(shift_blocks, batch_size)
    repeat_batch = [
        _apply_fixed_mlm_mask(row, tokenizer, mlm_probability, seed + 10_000 + index)
        for index, row in enumerate(repeat_raw)
    ]
    shift_batch = [
        _apply_fixed_mlm_mask(row, tokenizer, mlm_probability, seed + 20_000 + index)
        for index, row in enumerate(shift_raw)
    ]
    return repeat_batch, shift_batch


class RepeatThenShiftDataset(Dataset):
    """Sequential repeat → one shift → optional tail on the shift batch.

    Step schedule (0-based):

    - ``0 .. leading_repeat_steps-1``: ``repeat_batch``
    - ``leading_repeat_steps``: first ``shift_batch`` step
    - ``leading_repeat_steps+1 .. end``: ``shift_batch`` when ``tail_repeat_shift_steps > 0``

    Indices are ``step * batch_size + position_in_batch`` for ``SequentialSampler``.
    """

    def __init__(
        self,
        repeat_batch: list[dict[str, Any]],
        shift_batch: list[dict[str, Any]],
        num_steps: int | None = None,
        *,
        leading_repeat_steps: int | None = None,
        tail_repeat_shift_steps: int = 0,
    ) -> None:
        if len(repeat_batch) != len(shift_batch):
            msg = "repeat_batch and shift_batch must have the same length."
            raise ValueError(msg)
        if num_steps is not None and leading_repeat_steps is not None:
            msg = "Pass either num_steps or leading_repeat_steps, not both."
            raise ValueError(msg)
        if num_steps is not None:
            if num_steps < 2:
                msg = "num_steps must be at least 2 for a repeat-then-shift schedule."
                raise ValueError(msg)
            leading_repeat_steps = num_steps - 1
        if leading_repeat_steps is None or leading_repeat_steps < 1:
            msg = "leading_repeat_steps must be at least 1."
            raise ValueError(msg)
        if tail_repeat_shift_steps < 0:
            msg = "tail_repeat_shift_steps cannot be negative."
            raise ValueError(msg)
        self._repeat = repeat_batch
        self._shift = shift_batch
        self._batch_size = len(repeat_batch)
        self._leading = leading_repeat_steps
        self._tail = tail_repeat_shift_steps
        self._num_steps = leading_repeat_steps + 1 + tail_repeat_shift_steps

    @property
    def batch_size(self) -> int:
        """Sequences per optimizer step."""
        return self._batch_size

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def shift_step_index(self) -> int:
        """Optimizer step index (0-based) where the shift batch first appears."""
        return self._leading

    def __len__(self) -> int:
        return self._num_steps * self._batch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        step = index // self._batch_size
        in_batch = index % self._batch_size
        if step < self._leading:
            return self._repeat[in_batch]
        return self._shift[in_batch]
