"""CLI entrypoint for training-pathology experiments."""

from __future__ import annotations

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

_cache_root = _REPO_ROOT / ".cache"
_hf_home = os.environ.get("PATHOLOGIES_HF_HOME", str(_cache_root / "huggingface"))
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.environ.get(
    "PATHOLOGIES_HF_HUB_CACHE",
    str(Path(_hf_home) / "hub"),
)
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "PATHOLOGIES_HF_DATASETS_CACHE",
    str(_cache_root / "datasets"),
)
os.environ["TRANSFORMERS_CACHE"] = _hf_home

for _entry in os.environ.get("PYTHONPATH", str(_REPO_ROOT / "tutorials")).split(
    os.pathsep
):
    _entry = _entry.strip()
    if _entry and _entry not in sys.path:
        sys.path.insert(0, _entry)

import pathologies.experiments  # noqa: E402, F401 — register built-in experiments
from pathologies.experiment import RunConfig, available_experiments, get_experiment
from pathologies.experiments._common import (
    ABLATION_VARIANTS,
    BASELINE_MODE,
    BEST_PRACTICE_VIOLATIONS,
    FAILURE_MODES,
    TRAINING_PATHOLOGIES,
    pathology_category,
)
from pathologies.instrumentation.config import TelemetryConfig

logger = logging.getLogger(__name__)

_TELEMETRY_PRESETS = {
    "none": TelemetryConfig.none,
    "minimal": TelemetryConfig.minimal,
    "encoder": TelemetryConfig.encoder_mlm,
    "ablation": TelemetryConfig.ablation_transformer,
    "bilstm": TelemetryConfig.bilstm,
    "llm": TelemetryConfig.llm,
}


def main() -> None:
    """Parse CLI arguments and run the selected experiment."""
    parser = ArgumentParser(
        description="Run a HuggingFace Trainer / TRL training-pathology experiment.",
    )
    parser.add_argument(
        "-E",
        "--experiment",
        required=True,
        choices=available_experiments(),
        help="Registered experiment name.",
    )
    parser.add_argument(
        "-F",
        "--failure-mode",
        default="baseline",
        choices=sorted(FAILURE_MODES),
        help=(
            "Pathology or best-practice violation to induce (default: baseline). "
            f"Training: {', '.join(sorted(TRAINING_PATHOLOGIES))}. "
            f"Best practice: {', '.join(sorted(BEST_PRACTICE_VIOLATIONS))}."
        ),
    )
    parser.add_argument(
        "--variant",
        default="standard",
        choices=sorted(ABLATION_VARIANTS),
        help="Architecture variant for ablation_transformer (default: standard).",
    )
    parser.add_argument(
        "--telemetry",
        default=None,
        choices=sorted(_TELEMETRY_PRESETS),
        help="Telemetry preset override (default: experiment's built-in preset).",
    )
    parser.add_argument(
        "-C",
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory for Trainer checkpoints.",
    )
    parser.add_argument(
        "-L",
        "--log-dir",
        type=Path,
        required=True,
        help="W&B offline log directory (WANDB_DIR).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name (default: failure mode label).",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s] %(message)s (%(filename)s:%(lineno)s",
    )

    experiment = get_experiment(args.experiment)
    if args.failure_mode not in experiment.failure_modes:
        msg = (
            f"Experiment {args.experiment!r} does not support failure_mode "
            f"{args.failure_mode!r}"
        )
        raise SystemExit(msg)
    if experiment.variants and args.variant not in experiment.variants:
        msg = f"Experiment {args.experiment!r} requires --variant from {sorted(experiment.variants)}"
        raise SystemExit(msg)

    telemetry = (
        _TELEMETRY_PRESETS[args.telemetry]()
        if args.telemetry
        else experiment.default_telemetry()
    )

    if args.log_dir:
        os.environ.setdefault("WANDB_DIR", str(args.log_dir))

    config = RunConfig(
        output_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        seed=args.seed,
        failure_mode=args.failure_mode,
        variant=args.variant,
        telemetry=telemetry,
    )

    logger.info("Experiment: %s", experiment.name)
    logger.info("Description: %s", experiment.description)
    category = pathology_category(config.failure_mode)
    logger.info(
        "Failure mode: %s (%s) | variant: %s",
        config.failure_mode,
        category.value if category else BASELINE_MODE,
        config.variant if experiment.variants else "n/a",
    )
    logger.info("Output: %s", config.output_dir)
    logger.info("W&B dir: %s", config.log_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    experiment.run(config)


if __name__ == "__main__":
    main()
