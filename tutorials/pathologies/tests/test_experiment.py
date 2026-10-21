"""Registry tests for built-in telemetry experiments."""

import pathologies.experiments  # noqa: F401
from pathologies.experiment import available_experiments, get_experiment
from pathologies.experiments._common import ABLATION_VARIANTS, failure_modes_for_family


def test_experiment_names() -> None:
    """Five unified experiments should be registered."""
    assert available_experiments() == sorted(
        ["ablation_transformer", "bilstm", "encoder_mlm", "llm_grpo", "llm_sft"]
    )


def test_all_failure_modes_on_encoder() -> None:
    """Encoder experiment should accept every failure mode."""
    experiment = get_experiment("encoder_mlm")
    assert experiment.failure_modes == failure_modes_for_family("encoder")


def test_ablation_variants() -> None:
    """Ablation experiment exposes architecture variants."""
    experiment = get_experiment("ablation_transformer")
    assert experiment.variants == ABLATION_VARIANTS
