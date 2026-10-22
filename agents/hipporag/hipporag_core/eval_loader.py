"""Load toy eval fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from hipporag_core.models import OpenIEExtraction, Passage, Triple


def toy_multihop_dir() -> Path:
    """Path to ``eval/toy_multihop`` under the hipporag package."""
    return Path(__file__).resolve().parents[1] / "eval" / "toy_multihop"


def load_toy_corpus(path: Path | None = None) -> list[Passage]:
    """Load ``corpus.json`` passages.

    Args:
        path: Directory containing ``corpus.json`` (default: toy_multihop eval dir).

    Returns:
        Parsed passage list.
    """
    root = path or toy_multihop_dir()
    data = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    return [Passage(id=item["id"], text=item["text"]) for item in data]


def load_toy_openie_fixtures(path: Path) -> dict[str, OpenIEExtraction]:
    """Load ``openie.json`` from toy_multihop eval directory.

    Args:
        path: Directory containing ``openie.json``.

    Returns:
        Map passage id → extraction.
    """
    data = json.loads((path / "openie.json").read_text(encoding="utf-8"))
    result: dict[str, OpenIEExtraction] = {}
    for item in data:
        triples = [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                passage_id=item["passage_id"],
            )
            for t in item.get("triples", [])
        ]
        result[item["passage_id"]] = OpenIEExtraction(
            passage_id=item["passage_id"],
            entities=list(item.get("entities", [])),
            triples=triples,
        )
    return result
