# H-MEM implementation plan (qlab)

Reference: [2605.15701](https://arxiv.org/abs/2605.15701), PDF [`papers/2605.15701v1.pdf`](papers/2605.15701v1.pdf).  
Index semantics: [`HMEM_INDEX.md`](HMEM_INDEX.md).  
Upstream: [`UPSTREAM.md`](UPSTREAM.md) (no author repo; community [h-mem](https://github.com/Ardha-Eco-System/h-mem) for skim only).

## Delivery checklist

- [x] 1 Sources (paper PDF, upstream skim doc, README + PLAN Phase 0)
- [ ] 2 Shared platform (.env, build-image, optional Langfuse/Ollama)
- [x] 3 Reuse audit (below + module tree)
- [x] 4 Experiment core (`hmem_core/`, `server/`, config) — hybrid pipeline implemented
- [x] 5 Fixture path (`eval/toy_locomo_mini/`, bootstrap API, pytest) — 5 tests green
- [x] 6 LLM path (`HMEM_AGENTS=llm`, agents wired; no CI integration test yet)
- [x] 7 Runnable stack (compose, `_run_common.sh`, `run-agent.sh`) — agent-only
- [x] 8 Design doc (`HMEM_INDEX.md`)
- [x] 9 Register docs (`agents/README`, `REPOINDEX`, root README)
- [ ] 10 Commits (user-requested)

---

## Module map (qlab ← paper ← community reference)

| qlab module | Paper (§) | Community `src/hmem/…` | Implementation status |
|-------------|-----------|-------------------------|------------------------|
| `hmem_core/tree/temporal_semantic_tree.py` | 3.2 tree, α_l, β_l | `core/tree.py` | **Partial** — insert, embeddings, greedy α-merge |
| `hmem_core/graph/knowledge_graph.py` | 3.2 graph, profiles | `core/graph.py` | **Partial** — entities, fuzzy merge, multi-hop |
| `hmem_core/encoding/fragment_encoder.py` | embeddings | `utils/embeddings.py` | **Wraps** `BaseEncoder` |
| `hmem_core/indexing/event_extractor.py` | event per fragment | (in graph/tree builders) | **Fixture** + LLM protocol |
| `hmem_core/indexing/tree_builder.py` | incremental tree | `indexing/tree_builder.py` | **Done** — tree + encoder |
| `hmem_core/indexing/graph_builder.py` | entity/relation extract | `indexing/graph_builder.py` | **Partial** — fixture + LLM extractor |
| `hmem_core/indexing/indexer.py` | offline stage orchestrator | `indexing/indexer.py` | **Done** — `HybridIndexer` |
| `hmem_core/indexing/fixture_indexer.py` | — | — | **Done** (toy path) |
| `hmem_core/retrieval/planner.py` | 3.3 step 1 | `retrieval/planner.py` | **Fixture + LLM** |
| `hmem_core/retrieval/graph_search.py` | graph hop | `retrieval/graph_search.py` | **Partial** — seeds, hops, fragments |
| `hmem_core/retrieval/tree_search.py` | bottom-up tree | `retrieval/tree_search.py` | **Partial** — scope + filters |
| `hmem_core/retrieval/scoring.py` | Eq. 1–3, F score | `retrieval/reranker.py` (partial) | **Done** — S,T,R,F |
| `hmem_core/retrieval/engine.py` | online pipeline | `retrieval/engine.py` | **Done** — hybrid retrieve + optional gold fallback |
| `hmem_core/retrieval/synthesizer.py` | 3.3 generation | `retrieval/synthesizer.py` | **Fixture + LLM** |
| `hmem_core/retrieval/missing_info.py` | agentic follow-up | `retrieval/missing_info.py` | **Stub** |
| `hmem_core/agents/protocols.py` | LLM roles | `llm/adapter.py` | **Protocols** |
| `hmem_core/agents/fixture_agents.py` | test fixtures | smoke tests | **Partial** |
| `hmem_core/agents/llm_agents.py` | production LLM | `llm/openai.py` etc. | **Implemented** (env-gated) |
| `hmem_core/hybrid_store.py` | hybrid index | `engine.py` state | **Done** — tree + graph + fragments |
| `hmem_core/runtime.py` | — | `engine.py` | **Done** — encoder + engine refresh |

---

## Reuse audit (`agents_utils` — mandatory)

| Utility | H-MEM usage |
|---------|-------------|
| `memory.base_encoder.BaseEncoder` | All fragment/event/summary/query embeddings; tests use `HashDenseEncoder` / `FixedDenseEncoder` |
| `memory.dense_numpy.dense_cosine_similarity_numpy` | Semantic similarity S(m,Q) and tree merge α_l checks |
| `llm.openai_chat.create_chat_model` | Planner, summarizer, synthesizer when `HMEM_AGENTS=llm` |
| `env.load_agents_env`, `logging_setup`, `observability.langfuse_tracing` | Server lifespan (already in `main.py`) |
| `http.api_retry` | Optional: LangSearch/external embed APIs only (not core path) |

**Forbidden:** importing `hipporag_core`, `memgraphrag_core`, `cove_core`, or PyPI package `hmem`.

---

## Phased delivery

### Phase 1 — Scaffold (done + module stubs)

- [x] HTTP surface, toy eval, fixture indexer
- [x] Module tree with docstrings (`hmem_core/tree`, `graph`, `encoding`, `agents`, expanded `retrieval`)
- [x] Wire `POST /index` → `run_consolidation_pass` + `refresh_engine`

### Phase 2 — Offline indexing (paper §3.2)

- [x] `TemporalSemanticTree.insert_event` — embeddings, window bounds, greedy α merge
- [x] Summary generation hook (`SummarizerProtocol` — fixture concat + LLM)
- [x] `KnowledgeGraph` entity merge + fragment backlinks (heuristic)
- [x] `HybridIndexer.index_fragment` on corpus ingest
- [x] Unit tests: leaves + consolidation summary nodes

### Phase 3 — Online retrieval (paper §3.3)

- [x] `RetrievalPlanner.decompose(Q)` — fixture single sub-query; LLM JSON path
- [x] `GraphSearcher` expand + fragment linkage
- [x] `TreeSearcher.collect` with scope / fragment filter
- [x] Ranking — S, T, R, F with config weights
- [x] `RetrievalEngine.ask` — `/ask` route
- [x] `AnswerSynthesizer` — fixture heuristics + LLM

### Phase 4 — Eval (optional)

- [ ] `eval/loaders/` LoCoMo-mini slices; metrics F1 stub
- [ ] Compare against community harness **ideas** only (no vendored code)

---

## Paper fidelity checklist (review before marking “production-faithful”)

- [ ] Tree: exact paper incremental parent update (current: greedy community-style merge)
- [x] Graph: entity-centered links; multi-hop retrieval path exists
- [ ] Retrieval: SHORT/LONG/MIXED scopes materially change evidence (planner stub in fixture mode)
- [x] Ranking: combined semantic + temporal + robustness
- [x] No cross-experiment imports; encoders via `BaseEncoder` only

**Marked “testing”** when toy pytest passes end-to-end ([`IMPLEMENTATION_REVIEW.md`](IMPLEMENTATION_REVIEW.md)).
