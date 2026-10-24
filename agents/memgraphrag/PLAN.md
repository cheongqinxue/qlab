# MemGraphRAG implementation plan (qlab)

Reference: [2606.00610](https://arxiv.org/abs/2606.00610), [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG).

## Architecture (memory-first)

- **Primary API:** `MemGraphGlobalMemory` subclasses [`BaseMemory`](../agents_utils/memory/base_memory.py) (same contract as ReHarness `QdrantMemory`).
- **Structured state:** `ThreeLayerMemory` = paper M_ont / M_fac / M_pas (authoritative).
- **Indexing:** multi-agent loop (A_ext, A_det, A_res) + `indexing/graph_builder.py` → derived graph only.
- **Retrieval:** `MemGraphGlobalMemory.afind` → `retrieval/` (layer filter → PPR seeds → PPR).
- **Projections:** JSON (`storage/memory_store.py`), optional Neo4j (`storage/neo4j_store.py`).

The previous plan phases still apply; work is ordered **memory → agents → graph → retrieve**.

## Phase 0 — Scaffold

- [x] Directory tree: `memory/`, `agents/`, `indexing/`, `retrieval/`, `storage/`, routes, `eval/toy_conflict/`
- [x] `MemGraphGlobalMemory(BaseMemory)` + health counts via `ThreeLayerMemory`
- [x] `POST /reset` clears memory
- [ ] Skim official repo for field-level parity with upstream `Memory.py`

## Phase 1 — Global memory

- [x] Implement `add` / passage staging from `POST /corpus/passages`
- [x] `load_memory` / `save_memory` under `memgraphrag/.data/memory/`
- [x] Toy bootstrap from `eval/toy_conflict/` (`MEMGRAPHRAG_BOOTSTRAP_TOY=1`, `POST /corpus/bootstrap-toy`)

## Phase 2 — Multi-agent indexing

- [x] Wire `IndexOrchestrator` + fixture or LLM agents (`MEMGRAPHRAG_AGENTS`)
- [x] Hierarchical indexing graph + igraph PPR (`agents_utils/graph/ppr.py`)

## Phase 3 — Memory-aware retrieval

- [x] `MemoryGuidedRetriever` + `POST /retrieve` (dense fallback when graph empty)
- [ ] Optional `POST /ask` with LLM reader

## Phase 4 — Evaluation & observability

- [ ] Conflict toy demo + multi-hop questions
- [ ] Langfuse spans per agent role
- [ ] Compare to dense RAG and qlab `hipporag/`

## Non-goals (for now)

- Full KDD benchmark reproduction
- Vendoring the full official repository
