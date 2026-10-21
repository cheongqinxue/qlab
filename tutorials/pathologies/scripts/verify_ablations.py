"""Compare ablation variants — show why scaled attention, norm, and residuals matter."""

from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE = _REPO_ROOT / ".cache"
os.environ["HF_HOME"] = str(_CACHE / "huggingface")
os.environ["HF_HUB_CACHE"] = str(_CACHE / "huggingface" / "hub")
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))

import pathologies.experiments  # noqa: F401
from pathologies.experiment import RunConfig, get_experiment
from pathologies.experiments._common import ABLATION_VARIANTS
from pathologies.instrumentation.config import TelemetryConfig

METRIC_LINE = re.compile(r"'([^']+/[^']+)': '([^']+)'")


def _run_variant(variant: str, max_steps: int, max_samples: int) -> dict[str, float]:
    experiment = get_experiment("ablation_transformer")
    out = _REPO_ROOT / "tutorials/pathologies/.data" / f"ablation_{variant}"
    config = RunConfig(
        output_dir=out / "ckpt",
        log_dir=out / "tb",
        max_steps=max_steps,
        max_samples=max_samples,
        seed=42,
        failure_mode="baseline",
        variant=variant,
        telemetry=TelemetryConfig.ablation_transformer(),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        experiment.run(config)

    metrics: dict[str, list[float]] = {}
    for line in buffer.getvalue().splitlines():
        if "loss" not in line:
            continue
        for key, value in METRIC_LINE.findall(line):
            try:
                metrics.setdefault(key, []).append(float(value))
            except ValueError:
                continue

    summary: dict[str, float] = {}
    for key, values in metrics.items():
        summary[f"{key}_last"] = values[-1]
        summary[f"{key}_max"] = max(values)
    if "loss" in metrics:
        summary["loss_delta"] = metrics["loss"][ -1] - metrics["loss"][0]
    return summary


def main() -> None:
    max_steps = int(os.environ.get("VERIFY_MAX_STEPS", "30"))
    max_samples = int(os.environ.get("VERIFY_MAX_SAMPLES", "256"))
    report = {
        variant: _run_variant(variant, max_steps, max_samples)
        for variant in sorted(ABLATION_VARIANTS)
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
