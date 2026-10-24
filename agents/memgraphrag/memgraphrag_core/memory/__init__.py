"""Global three-layer memory and ``BaseMemory`` adapter for MemGraphRAG.

The paper's M_ont / M_fac / M_pas live in ``ThreeLayerMemory``. ``MemGraphGlobalMemory``
subclasses ``agents_utils.memory.base_memory.BaseMemory`` so the experiment shares the
same memory-bank contract as ReHarness (``QdrantMemory``), while adding graph-specific
indexing and retrieval orchestration elsewhere in ``memgraphrag_core``.
"""

from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.memory.three_layer import ThreeLayerMemory

__all__ = ["MemGraphGlobalMemory", "ThreeLayerMemory"]
