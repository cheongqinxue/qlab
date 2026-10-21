"""Parse local ``run-*.wandb`` files into :class:`~bev.ingest.types.HistoryEvent`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2

from bev.ingest.constants import STEP_FIELD_KEYS, SYSTEM_HISTORY_KEYS
from bev.ingest.types import HistoryEvent, RunInfo
from bev.ingest.wandb_transaction_log import WandbTransactionLog


class WandbRunReader:
    """Read history metrics from a W&B offline/online run transaction log."""

    def __init__(self, wandb_file: Path) -> None:
        """Bind to a ``run-<id>.wandb`` file path."""
        self._wandb_file = Path(wandb_file)

    @property
    def wandb_file(self) -> Path:
        return self._wandb_file

    def iter_history(
        self,
        *,
        start_offset: int = 0,
    ) -> Iterator[tuple[int, HistoryEvent]]:
        """Yield ``(byte_offset_after_record, event)`` for each history record.

        Args:
            start_offset: Resume scanning after this byte offset (0 = from header).
        """
        log = WandbTransactionLog()
        log.open_for_scan(str(self._wandb_file))
        if start_offset:
            log.seek(start_offset)
        while True:
            before = log.byte_offset
            payload = log.scan_data()
            if payload is None:
                break
            after = log.byte_offset
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.WhichOneof("record_type") != "history":
                continue
            event = self._history_from_record(record)
            if event is not None:
                yield after, event
        log.close()

    def read_run_info(self) -> RunInfo | None:
        """Return run metadata if a ``RunRecord`` is present in the log."""
        log = WandbTransactionLog()
        log.open_for_scan(str(self._wandb_file))
        found: RunInfo | None = None
        while True:
            payload = log.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.WhichOneof("record_type") != "run":
                continue
            run = record.run
            found = RunInfo(
                wandb_run_id=run.run_id,
                display_name=run.display_name or "",
                project=run.project or "",
                entity=run.entity or "",
            )
            break
        log.close()
        return found

    def _history_from_record(self, record: wandb_internal_pb2.Record) -> HistoryEvent | None:
        raw: dict[str, Any] = {}
        for item in record.history.item:
            key = _history_item_key(item)
            if not key:
                continue
            raw[key] = json.loads(item.value_json)
        step = raw.get("_step")
        if step is None:
            return None
        metrics: dict[str, Any] = {}
        global_step: int | None = None
        epoch: float | None = None
        logged_at: float | None = raw.get("_timestamp")
        for key, value in raw.items():
            if key in SYSTEM_HISTORY_KEYS:
                continue
            if key == "train/global_step" and isinstance(value, (int, float)):
                global_step = int(value)
                continue
            if key == "epoch" and isinstance(value, (int, float)):
                epoch = float(value)
                continue
            if key in STEP_FIELD_KEYS:
                continue
            metrics[key] = value
        return HistoryEvent(
            step=int(step),
            global_step=global_step,
            epoch=epoch,
            logged_at=float(logged_at) if logged_at is not None else None,
            metrics=metrics,
        )


def _history_item_key(item: wandb_internal_pb2.HistoryItem) -> str:
    """Resolve metric key from W&B 0.22 ``key`` or 0.30+ ``nested_key``."""
    if item.key:
        return item.key
    if item.nested_key:
        return "/".join(item.nested_key)
    return ""
