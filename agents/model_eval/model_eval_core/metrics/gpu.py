"""GPU VRAM sampling for local Ollama eval runs (nvidia-smi + optional Ollama /api/ps)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuVramSample:
    """VRAM usage for one GPU at one instant."""

    gpu_index: int
    used_mib: int
    total_mib: int


@dataclass
class CaseGpuMetrics:
    """VRAM readings for a single eval case."""

    gpu_index: int = 0
    used_mib_before: int | None = None
    used_mib_after: int | None = None
    used_mib_peak: int | None = None
    total_mib: int | None = None
    ollama_loaded_vram_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for responses.jsonl."""
        delta: int | None = None
        if self.used_mib_before is not None and self.used_mib_after is not None:
            delta = self.used_mib_after - self.used_mib_before
        return {
            "gpu_index": self.gpu_index,
            "used_mib": {
                "before": self.used_mib_before,
                "after": self.used_mib_after,
                "peak": self.used_mib_peak,
                "delta": delta,
            },
            "total_mib": self.total_mib,
            "ollama_loaded_vram_bytes": self.ollama_loaded_vram_bytes,
        }


@dataclass
class RuntimeStats:
    """Aggregated latencies for a full model folder run."""

    wall_s: float = 0.0
    case_latencies_s: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize summary.json runtime block."""
        lat = self.case_latencies_s
        if not lat:
            return {"wall_s": round(self.wall_s, 3), "case_latency_s": {}}
        sorted_lat = sorted(lat)
        n = len(sorted_lat)
        p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
        return {
            "wall_s": round(self.wall_s, 3),
            "case_latency_s": {
                "mean": round(sum(lat) / n, 3),
                "min": round(min(lat), 3),
                "max": round(max(lat), 3),
                "p50": round(sorted_lat[n // 2], 3),
                "p95": round(sorted_lat[p95_idx], 3),
                "sum": round(sum(lat), 3),
            },
        }


def eval_gpu_index() -> int:
    """GPU index for VRAM queries (``MODEL_EVAL_GPU_INDEX``, default 0)."""
    raw = os.environ.get("MODEL_EVAL_GPU_INDEX", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def query_gpu_vram_mib(gpu_index: int | None = None) -> GpuVramSample | None:
    """Read used/total VRAM via ``nvidia-smi``.

    Args:
        gpu_index: CUDA device index; defaults to ``eval_gpu_index()``.

    Returns:
        Sample or None when nvidia-smi is unavailable or fails.
    """
    idx = eval_gpu_index() if gpu_index is None else gpu_index
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={idx}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [p.strip() for p in proc.stdout.strip().split(",")]
    if len(parts) != 2:
        return None
    try:
        used_mib = int(float(parts[0]))
        total_mib = int(float(parts[1]))
    except ValueError:
        return None
    return GpuVramSample(gpu_index=idx, used_mib=used_mib, total_mib=total_mib)


def ollama_api_base(openai_base_url: str | None = None) -> str:
    """Derive Ollama root URL from the OpenAI-compatible base URL."""
    base = openai_base_url or os.environ.get(
        "OLLAMA_OPENAI_BASE_URL",
        "http://127.0.0.1:11434/v1",
    )
    return base.rstrip("/").removesuffix("/v1")


def query_ollama_loaded_vram_bytes(api_base: str | None = None) -> int | None:
    """Sum ``size_vram`` for models listed in Ollama ``GET /api/ps``.

    Args:
        api_base: Ollama host root (no ``/v1`` suffix).

    Returns:
        Total loaded VRAM in bytes, or None if the API is unreachable.
    """
    root = api_base or ollama_api_base()
    url = f"{root}/api/ps"
    try:
        with urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.debug("Ollama /api/ps unavailable: %s", exc)
        return None
    models = payload.get("models") or []
    total = 0
    for entry in models:
        size = entry.get("size_vram")
        if isinstance(size, int | float):
            total += int(size)
    return total


@asynccontextmanager
async def track_vram_peak(
    gpu_index: int | None = None,
    *,
    poll_interval_s: float = 0.25,
) -> AsyncIterator[list[int]]:
    """Poll GPU used-MiB until the context exits; yield a one-element peak list.

    Args:
        gpu_index: CUDA device index.
        poll_interval_s: Seconds between nvidia-smi polls during the case.

    Yields:
        Mutable list ``[peak_used_mib]`` updated by the background poller.
    """
    idx = eval_gpu_index() if gpu_index is None else gpu_index
    peak_holder: list[int] = [0]

    def _note(sample: GpuVramSample | None) -> None:
        if sample is not None:
            peak_holder[0] = max(peak_holder[0], sample.used_mib)

    stop = asyncio.Event()

    async def _poll_loop() -> None:
        while not stop.is_set():
            _note(query_gpu_vram_mib(idx))
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
            except TimeoutError:
                continue

    _note(query_gpu_vram_mib(idx))
    task = asyncio.create_task(_poll_loop())
    try:
        yield peak_holder
    finally:
        stop.set()
        await task
        _note(query_gpu_vram_mib(idx))
