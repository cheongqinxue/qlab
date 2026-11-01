"""Backward-compatible import surface — use :mod:`prompts` instead."""

from model_eval_core.prompts import CuratedPrompt, read_curated_jsonl

__all__ = ["CuratedPrompt", "read_curated_jsonl"]
