"""Pathology taxonomy and profile resolution tests."""

import pytest

from pathologies.experiments._common import (
    BASELINE_MODE,
    BEST_PRACTICE_VIOLATIONS,
    PathologyCategory,
    TRAINING_PATHOLOGIES,
    effective_max_steps,
    failure_modes_for_family,
    pathology_category,
    resolve_failure_profile,
)


def test_pathology_categories() -> None:
    """Every non-baseline mode belongs to exactly one taxonomy bucket."""
    assert pathology_category(BASELINE_MODE) is None
    for mode in TRAINING_PATHOLOGIES:
        assert pathology_category(mode) is PathologyCategory.TRAINING
    for mode in BEST_PRACTICE_VIOLATIONS:
        assert pathology_category(mode) is PathologyCategory.BEST_PRACTICE


def test_encoder_supports_all_encoder_modes() -> None:
    """Encoder MLM should expose universal, encoder, and best-practice modes."""
    modes = failure_modes_for_family("encoder")
    assert BEST_PRACTICE_VIOLATIONS.issubset(modes)
    assert "attention_sink" in modes
    assert "bad_mlm_mask" in modes


def test_bilstm_excludes_encoder_only_modes() -> None:
    """BiLSTM should not expose MLM-only best-practice violations."""
    modes = failure_modes_for_family("bilstm")
    assert "bad_mlm_mask" not in modes
    assert "logit_collapse" in modes


def test_effective_max_steps_uses_profile_minimum() -> None:
    """Profiles with long runs should bump short CLI step budgets."""
    profile = resolve_failure_profile("encoder", "rank_collapse")
    assert effective_max_steps(30, profile) == profile.min_steps


def test_unknown_mode_raises() -> None:
    """Unknown failure modes should fail fast."""
    with pytest.raises(KeyError):
        resolve_failure_profile("encoder", "not_a_mode")
