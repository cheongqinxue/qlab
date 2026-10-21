# MemGraphRAG implementation plan (qlab)

Reference: [2606.00610](https://arxiv.org/abs/2606.00610), [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG).

## Phase 0 — Scaffold (current)

- [x] Experiment folder, Docker, health API, docs
- [ ] Skim official repo for memory schema and agent roles

## Phase 1 — Global memory

- [ ] `MemGraphRAGMemory`: ontology (`M_ont`), facts (`M_fac`), passages (`M_pas`)
- [ ] Persist under `memgraphrag/.data/memory/`
- [ ] Ingest API: add passages, list layer counts

## Phase 2 — Multi-agent indexing

- [ ] Extraction agent → facts + ontology updates
- [ ] Conflict detection agent (hard conflicts vs passage evidence)
- [ ] Resolution agent (connected conflict groups)
- [ ] Build **hierarchical indexing graph** from memory

## Phase 3 — Memory-aware retrieval

- [ ] Memory filtering + structure-aware PPR seeds
- [ ] `POST /retrieve` → ranked passages / subgraph context
- [ ] Optional answer generation endpoint

## Phase 4 — Evaluation & observability

- [ ] Mini corpus + multi-hop questions
- [ ] Langfuse traces per agent role
- [ ] Compare to dense RAG and qlab `hipporag/` scaffold

## Non-goals (for now)

- Full KDD benchmark reproduction
- Vendoring the full official repository
