"""Pluggable training telemetry for HuggingFace Trainer."""

from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import (
    TelemetryHandle,
    attach_telemetry,
    build_trainer_callbacks,
)

__all__ = [
    "TelemetryConfig",
    "TelemetryHandle",
    "attach_telemetry",
    "build_trainer_callbacks",
]
