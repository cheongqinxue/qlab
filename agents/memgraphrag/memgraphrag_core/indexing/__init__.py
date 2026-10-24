"""Offline indexing: multi-agent loop + hierarchical graph build.

Indexing **does not** replace ``MemGraphGlobalMemory``; it populates ``ThreeLayerMemory``
and builds a derived ``IndexingGraph`` (in-memory + optional Neo4j) used only at query time.
"""
