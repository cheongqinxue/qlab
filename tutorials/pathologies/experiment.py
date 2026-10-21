"""Experiment registry for Trainer-based telemetry runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pathologies.instrumentation.config import TelemetryConfig

_EXPERIMENTS: dict[str, Experiment] = {}


@dataclass(frozen=True)
class RunConfig:
    """Runtime paths and options shared by all experiments."""

    output_dir: Path
    log_dir: Path
    run_name: str | None = None
    max_steps: int = 100
    max_samples: int = 512
    seed: int = 42
    failure_mode: str = "baseline"
    variant: str = "standard"
    telemetry: TelemetryConfig | None = None


@dataclass(frozen=True)
class Experiment:
    """A named training scenario; pathologies selected via ``RunConfig.failure_mode``."""

    name: str
    description: str
    model_family: str
    training_kind: str
    profile_family: str
    failure_modes: frozenset[str]
    run: Callable[[RunConfig], None]
    variants: frozenset[str] = frozenset()
    default_telemetry: Callable[[], TelemetryConfig] = field(
        default=TelemetryConfig.minimal
    )


def register_experiment(experiment: Experiment) -> Experiment:
    """Register an experiment under its ``name``."""
    if experiment.name in _EXPERIMENTS:
        msg = f"Experiment already registered: {experiment.name}"
        raise ValueError(msg)

    original_run = experiment.run

    def run_with_seed(config: RunConfig) -> None:
        from pathologies.experiments._common import set_run_seed

        set_run_seed(config.seed)
        original_run(config)

    seeded = Experiment(
        name=experiment.name,
        description=experiment.description,
        model_family=experiment.model_family,
        training_kind=experiment.training_kind,
        profile_family=experiment.profile_family,
        failure_modes=experiment.failure_modes,
        run=run_with_seed,
        variants=experiment.variants,
        default_telemetry=experiment.default_telemetry,
    )
    _EXPERIMENTS[experiment.name] = seeded
    return seeded


def get_experiment(name: str) -> Experiment:
    """Look up a registered experiment by name."""
    if name not in _EXPERIMENTS:
        available = ", ".join(sorted(_EXPERIMENTS)) or "(none)"
        msg = f"Unknown experiment {name!r}. Available: {available}"
        raise KeyError(msg)
    return _EXPERIMENTS[name]


def available_experiments() -> list[str]:
    """Return registered experiment names."""
    return sorted(_EXPERIMENTS)
