"""Training pathologies lab — HuggingFace Trainer, TRL, and Weights & Biases."""

from pathologies.experiment import Experiment, RunConfig, available_experiments, get_experiment
from pathologies.instrumentation import TelemetryConfig, TelemetryHandle, attach_telemetry, build_trainer_callbacks

__all__ = [
    "Experiment",
    "RunConfig",
    "TelemetryConfig",
    "TelemetryHandle",
    "attach_telemetry",
    "available_experiments",
    "build_trainer_callbacks",
    "get_experiment",
]
