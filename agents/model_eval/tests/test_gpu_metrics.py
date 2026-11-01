"""GPU metrics helpers."""

from __future__ import annotations

from model_eval_core.metrics import RuntimeStats, query_gpu_vram_mib


class TestRuntimeStats:
    """Suite-level latency aggregation."""

    def test_as_dict_includes_percentiles(self) -> None:
        stats = RuntimeStats(wall_s=100.0, case_latencies_s=[1.0, 2.0, 10.0, 4.0])
        block = stats.as_dict()
        assert block["wall_s"] == 100.0
        lat = block["case_latency_s"]
        assert lat["mean"] == 4.25
        assert lat["p50"] == 4.0
        assert lat["max"] == 10.0


class TestQueryGpuVram:
    """nvidia-smi integration when present."""

    def test_query_returns_sample_or_none(self) -> None:
        sample = query_gpu_vram_mib(0)
        if sample is not None:
            assert sample.used_mib >= 0
            assert sample.total_mib > 0
            assert sample.used_mib <= sample.total_mib
