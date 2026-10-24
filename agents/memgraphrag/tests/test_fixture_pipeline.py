"""End-to-end fixture agent flow on toy conflict data."""

from __future__ import annotations

import json
from pathlib import Path

from memgraphrag_core.agents.conflict_detect import FixtureConflictDetectionAgent
from memgraphrag_core.agents.conflict_resolve import FixtureConflictResolutionAgent
from memgraphrag_core.agents.extraction import FixtureExtractionAgent
from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


class TestFixturePipeline:
    """Load eval JSON and run detect → resolve."""

    @staticmethod
    def eval_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "eval" / "toy_conflict"

    def test_newton_conflict_resolution(self) -> None:
        root = self.eval_dir()
        corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
        extraction = json.loads((root / "extraction.json").read_text(encoding="utf-8"))
        conflicts = json.loads((root / "conflicts.json").read_text(encoding="utf-8"))
        resolutions = json.loads((root / "resolutions.json").read_text(encoding="utf-8"))

        memory = ThreeLayerMemory()
        for row in corpus["passages"]:
            memory.add_passage(PassageRecord(id=row["id"], text=row["text"]))

        ext_agent = FixtureExtractionAgent(extraction)
        for pid in extraction:
            payload = ext_agent.extract(pid, "")
            for schema in payload.get("schemas", []):
                memory.add_ontology(
                    OntologySchemaRecord(
                        id=schema["id"],
                        label=schema["label"],
                        extraction_count=1,
                    )
                )
            for fact in payload.get("facts", []):
                memory.add_fact(
                    FactRecord(
                        id=fact["id"],
                        text=fact["text"],
                        passage_ids=fact.get("passage_ids", []),
                    )
                )

        assert memory.fact_count == 2
        detect = FixtureConflictDetectionAgent(conflicts["groups"])
        groups = detect.detect(memory, new_fact_ids=["f_newton_1643"])
        assert len(groups) == 1

        resolve = FixtureConflictResolutionAgent(resolutions)
        summary = resolve.resolve(memory, groups[0])
        assert "f_newton_1643" in summary["dropped_fact_ids"]
        assert memory.fact_count == 1
