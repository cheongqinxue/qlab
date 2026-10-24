"""Memory-guided retrieval (paper §4.3).

``MemGraphGlobalMemory.afind`` delegates here: layer similarity filter → PPR seeds → PPR
→ ranked passages. Dense vector fallback when graph coverage is insufficient.
"""
