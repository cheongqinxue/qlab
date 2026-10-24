# H-MEM hybrid index (design)

Authoritative semantics for qlab’s H-MEM reimplementation. **Logic follows** [arXiv:2605.15701](https://arxiv.org/abs/2605.15701); **code layout** informed by community [h-mem](https://github.com/Ardha-Eco-System/h-mem) (reference only).

## Implemented (scaffold / fixture)

| Piece | Status | Module |
|-------|--------|--------|
| Fragment ingest API | Done | `server/routes/corpus.py` |
| Fixture leaves + flat edges | Done | `indexing/fixture_indexer.py` |
| Placeholder consolidation counter | Done | `indexing/fixture_indexer.run_consolidation_pass` |
| Fixture QA | Done | `retrieval/fixture_retrieval.py` |

## Planned (paper-faithful)

### Offline indexing (§3.2)

1. **Events** — Extract fine-grained semantic events from each fragment `f_i` (LLM or fixture JSON).
2. **Tree** — For each new leaf at level `l=0..L-1`:
   - Assign **temporal window** `β_l` by timestamp.
   - Among nodes in the same window, if similarity ≥ **α_l**, merge into cluster; else new parent.
   - Parent **summary** consolidates children (LLM or fixture string).
3. **Graph** — Extract entities/relations; **disambiguate** (normalize, overlap, fuzzy match); link entities ↔ source fragments; optional **profiles** for salient types.
4. **Vectors** — Embed fragments, events, summaries via **`agents_utils.memory.base_encoder.BaseEncoder`**.

### Online retrieval (§3.3)

1. **Planning** — Decompose `Q` into `{Q_k}` with dependencies; per sub-query infer **scope** ∈ {SHORT, LONG, MIXED} and temporal hints.
2. **Graph** — Seed entities from `Q_k` (lexical + vector match); **multi-hop** expansion.
3. **Tree** — Map graph hits → fragments → leaf events; **bottom-up** search per scope.
4. **Ranking** — Score each evidence `m`:  
   `F(m,Q_k,t) = θ₁ S(m,Q_k) + θ₂ T(m,Q_k) + θ₃ R(m,t)`  
   with `S` = embedding cosine (`dense_numpy`), `T` = temporal overlap/distance (paper Eq. 1), `R` = robustness / forgetting (paper Eq. 2).
5. **Generation** — Sub-answers per `Q_k`; synthesize with prerequisite context (LLM via `agents_utils.llm`).

## qlab `agents_utils` binding

| Paper need | qlab utility |
|------------|----------------|
| Embeddings | `BaseEncoder`, `dense_cosine_similarity_numpy` |
| LLM planner/synth | `agents_utils.llm.openai_chat.create_chat_model` |
| Env / logs / traces | `env`, `logging_setup`, `langfuse_tracing` |
| Retries (external APIs) | `agents_utils.http.api_retry` |

**Not used:** Hippo/MemGraph PPR, Qdrant, Neo4j (in-process graph + tree unless scale doc revisits).

## Persistence

v0: in-memory (`hmem_core/hybrid_store.py`). Later: serialize tree + graph under `config.memory_data_dir` (community `TreeBuilder` JSON pattern as reference).
