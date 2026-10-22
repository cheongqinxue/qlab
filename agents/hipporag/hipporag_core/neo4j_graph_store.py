"""Neo4j-backed graph store (dual-writes to memory for PPR export)."""

from __future__ import annotations

import os
from typing import Any

from neo4j import Driver, GraphDatabase

from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.phrase_encoder import PhraseEncoder


class Neo4jGraphStore(MemoryGraphStore):
    """Persist MERGE operations to Neo4j Browser while indexing in RAM."""

    def __init__(self, driver: Driver) -> None:
        """Attach Neo4j driver to in-memory index."""
        super().__init__()
        self._driver = driver

    @classmethod
    def from_env(cls) -> Neo4jGraphStore:
        """Connect using ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD``."""
        uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "change-me-neo4j")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver)

    def close(self) -> None:
        """Close driver."""
        self._driver.close()

    def _run(self, query: str, **params: Any) -> None:
        with self._driver.session() as session:
            session.run(query, **params)

    def stage_passage(self, passage) -> None:  # noqa: ANN001 — Passage model
        """Stage passage and MERGE ``Passage`` node."""
        super().stage_passage(passage)
        from hipporag_core.content_hash import passage_content_hash

        self._run(
            """
            MERGE (p:Passage {id: $id})
            SET p.text = $text, p.content_hash = $hash
            """,
            id=passage.id,
            text=passage.text,
            hash=passage_content_hash(passage.text),
        )

    def merge_entity(self, phrase: str, encoder: PhraseEncoder) -> str:
        """Merge entity in RAM and Neo4j."""
        eid = super().merge_entity(phrase, encoder)
        text = self._entities[eid]
        self._run(
            """
            MERGE (e:Entity {id: $id})
            SET e.text = $text
            """,
            id=eid,
            text=text,
        )
        return eid

    def add_mention(self, entity_eid: str, passage_id: str, count: int = 1) -> None:
        """Increment mentions in RAM and Neo4j."""
        super().add_mention(entity_eid, passage_id, count)
        self._run(
            """
            MATCH (e:Entity {id: $eid}), (p:Passage {id: $pid})
            MERGE (e)-[m:MENTIONS]->(p)
            SET m.count = coalesce(m.count, 0) + $count
            """,
            eid=entity_eid,
            pid=passage_id,
            count=count,
        )

    def add_rel(self, subject_eid: str, object_eid: str) -> None:
        """Add REL edge in RAM and Neo4j."""
        super().add_rel(subject_eid, object_eid)
        self._run(
            """
            MATCH (a:Entity {id: $sub}), (b:Entity {id: $obj})
            MERGE (a)-[r:REL]->(b)
            SET r.weight = coalesce(r.weight, 1.0)
            """,
            sub=subject_eid,
            obj=object_eid,
        )

    def add_synonym(self, eid_a: str, eid_b: str) -> None:
        """Add SYNONYM edge in RAM and Neo4j."""
        super().add_synonym(eid_a, eid_b)
        self._run(
            """
            MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
            MERGE (a)-[s:SYNONYM]-(b)
            SET s.score = coalesce(s.score, 1.0)
            """,
            a=eid_a,
            b=eid_b,
        )

    def clear(self) -> None:
        """Clear RAM and detach-delete graph in Neo4j."""
        super().clear()
        self._run("MATCH (n) DETACH DELETE n")
