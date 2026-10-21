"""Tests for local ``.wandb`` parsing."""

from bev.ingest.wandb_reader import WandbRunReader


class TestWandbRunReader:
    def test_read_run_info(self, minimal_wandb_run) -> None:
        reader = WandbRunReader(minimal_wandb_run)
        info = reader.read_run_info()
        assert info is not None
        assert info.wandb_run_id == "bevminimal1"
        assert info.project == "bev_test"

    def test_iter_history_yields_metrics(self, minimal_wandb_run) -> None:
        reader = WandbRunReader(minimal_wandb_run)
        events = [event for _, event in reader.iter_history()]
        assert len(events) == 2
        assert events[0].metrics["train/loss"] == 1.0
        assert events[1].metrics["train/grad_norm/layer.1"] == 0.1

    def test_incremental_offset_skips_replayed_history(self, minimal_wandb_run) -> None:
        reader = WandbRunReader(minimal_wandb_run)
        offsets = [offset for offset, _ in reader.iter_history()]
        assert len(offsets) == 2
        resume_offset = offsets[0]
        resumed = [event for _, event in reader.iter_history(start_offset=resume_offset)]
        assert len(resumed) == 1
