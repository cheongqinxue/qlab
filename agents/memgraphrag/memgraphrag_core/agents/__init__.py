"""Multi-agent group A = {A_ext, A_det, A_res} (paper §4.1).

Each agent reads and writes ``ThreeLayerMemory`` through ``MemGraphGlobalMemory.layers``.
Implementations may use LLM structured output or committed JSON fixtures (default for toy eval).
"""
