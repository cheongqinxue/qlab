"""Ingest constants shared with the metric catalog."""

SYSTEM_HISTORY_KEYS: frozenset[str] = frozenset(
    {
        "_step",
        "_runtime",
        "_timestamp",
    }
)

STEP_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "train/global_step",
        "epoch",
    }
)
