"""Neo4j projection of three-layer memory and indexing graph."""

from __future__ import annotations

import os

from agents_utils.neo4j_env import neo4j_password_from_env

from memgraphrag_core.indexing.graph_builder import IndexingGraph
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


class Neo4jMemoryProjection:
    """Dual-write helper for Browser visualization."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Store connection parameters (driver created lazily)."""
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: object | None = None

    @classmethod
    def from_env(cls) -> Neo4jMemoryProjection:
        """Connect using ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD``."""
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = neo4j_password_from_env()
        return cls(uri, user, password)

    def _get_driver(self):  # noqa: ANN202 — neo4j Driver
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    def _run(self, query: str, **params: object) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run(query, **params)

    def sync_layers(
        self,
        memory: ThreeLayerMemory,
        graph: IndexingGraph | None = None,
    ) -> None:
        """MERGE nodes and relationships from current memory tiers."""
        for schema in memory.iter_ontology():
            self._run(
                """
                MERGE (s:Schema {id: $id})
                SET s.label = $label, s.extraction_count = $count
                """,
                id=schema.id,
                label=schema.label,
                count=schema.extraction_count,
            )
        for passage in memory.iter_passages():
            self._run(
                """
                MERGE (p:Passage {id: $id})
                SET p.text = $text
                """,
                id=passage.id,
                text=passage.text,
            )
        for fact in memory.iter_facts():
            self._run(
                """
                MERGE (f:Fact {id: $id})
                SET f.text = $text
                """,
                id=fact.id,
                text=fact.text,
            )
            for pid in fact.passage_ids:
                self._run(
                    """
                    MATCH (f:Fact {id: $fid}), (p:Passage {id: $pid})
                    MERGE (f)-[:SUPPORTED_BY]->(p)
                    """,
                    fid=fact.id,
                    pid=pid,
                )
            for sid in fact.schema_ids:
                self._run(
                    """
                    MATCH (f:Fact {id: $fid}), (s:Schema {id: $sid})
                    MERGE (f)-[:INSTANCE_OF]->(s)
                    """,
                    fid=fact.id,
                    sid=sid,
                )
        _ = graph

    def clear(self) -> None:
        """Remove all MemGraphRAG nodes from Neo4j."""
        self._run("MATCH (n) WHERE n:Schema OR n:Fact OR n:Passage DETACH DELETE n")

    def close(self) -> None:
        """Release driver resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
