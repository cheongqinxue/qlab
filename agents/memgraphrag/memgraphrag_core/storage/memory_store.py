"""JSON persistence for ``ThreeLayerMemory`` (``memory.json``)."""

from __future__ import annotations

import json
from pathlib import Path

from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


def memory_json_path(data_dir: str | Path) -> Path:
    """Canonical snapshot path."""
    return Path(data_dir) / "memory.json"


def save_memory(memory: ThreeLayerMemory, data_dir: str | Path) -> Path:
    """Serialize all tiers to disk."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = memory_json_path(root)
    payload = {
        "ontology": [o.model_dump() for o in memory.iter_ontology()],
        "facts": [f.model_dump() for f in memory.iter_facts()],
        "passages": [p.model_dump() for p in memory.iter_passages()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_memory(data_dir: str | Path) -> ThreeLayerMemory:
    """Restore tiers from ``memory.json`` if present."""
    path = memory_json_path(data_dir)
    memory = ThreeLayerMemory()
    if not path.is_file():
        return memory
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("passages", []):
        memory.add_passage(PassageRecord.model_validate(row))
    for row in data.get("ontology", []):
        memory.add_ontology(OntologySchemaRecord.model_validate(row))
    for row in data.get("facts", []):
        memory.add_fact(FactRecord.model_validate(row))
    return memory
