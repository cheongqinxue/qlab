# H-MEM implementation review (agents_utils + paper fidelity)

Date: 2026-10-25. Status: **testing** (qlab harness bar met) — hybrid pipeline + toy pytest green; **not** production-faithful to every paper detail or benchmark numbers. See [`PLAN.md`](PLAN.md) fidelity checklist and Phase 4.

This document records **what** differs from the paper and **why** those gaps were accepted, so a future pass can reopen items deliberately.

---

## Scope decision (acceptable level for v0)

**In scope (done):**

- Experiment shell: FastAPI, Docker/run scripts, import wall, `agents_utils` wiring.
- Hybrid store: temporal tree + entity graph + fragment list in memory.
- Offline path: ingest → optional consolidation → refresh engine.
- Online path: plan → graph → tree → rank (S/T/R) → synthesize; `/ask` wired.
- Default **`HMEM_AGENTS=fixture`**: deterministic, keyless pytest (5 tests).
- Optional **`HMEM_AGENTS=llm`**: planner, summarizer, extractor, synthesizer classes (manual / env-gated).

**Explicitly out of scope for v0:**

- Reproducing paper tables on LoCoMo / LongMemEval / REALTALK.
- Bit-exact tree incremental update vs author or community reference.
- LLM path as CI default or recorded golden traces.
- On-disk persistence under `memory_data_dir` (design only in [`HMEM_INDEX.md`](HMEM_INDEX.md)).

**Success criterion used:** same bar as sibling harnesses — runnable demo + fixture tests — not arXiv replication.

---

## Why gaps were deferred (rationale)

| Driver | Implication for H-MEM |
|--------|------------------------|
| **qlab harness-first workflow** | Ship module boundaries, API, and pytest before tuning α/β/θ on real benchmarks. |
| **No official author code** ([`UPSTREAM.md`](UPSTREAM.md)) | Tree merge and planner JSON inferred from PDF + community skim; heuristics preferred over guessing author behavior without golden tests. |
| **Fixture mode = CI contract** | Tests use `HashDenseEncoder` and fixture agents; avoids API keys and flakiness. LLM code exists but is not proof of paper fidelity. |
| **Phase 4 eval marked optional** | Without LoCoMo-mini loader + metrics, there is no feedback loop to justify “full paper” retrieval tuning in-repo. |
| **Tree consolidation risk** | Incremental parent updates are easy to misimplement; greedy merge + β windows gives working summaries with unit coverage. |
| **Toy QA plumbing** | Keyword overlap on semantic score and `FixtureSynthesizer` heuristics unblock hybrid retrieval tests without gold `fixture_answer`; these are **harness aids**, not methodology. |
| **User sign-off (2026-10-25)** | Current depth judged sufficient; deeper fidelity deferred until explicitly requested. |

When revisiting, pick an axis (tree / planner / eval / LLM CI) and add **golden tests or benchmark slices** first — otherwise changes are unmeasurable.

---

## 1. `agents_utils` usage

| Requirement | Status | Where |
|-------------|--------|--------|
| `BaseEncoder` for all embeddings | OK | `encoding/fragment_encoder.py`, `HybridIndexer`, tree insert |
| `dense_cosine_similarity_numpy` for S(m,Q) | OK | `retrieval/scoring.py`, tree α-merge |
| `llm.openai_chat` for LLM agents | OK | `agents/llm_agents.py` (planner, summarizer, extractor, synthesizer) |
| `env`, `logging_setup`, `langfuse_tracing` | OK | `main.py`, `server/settings.py` |
| No cross-experiment imports | OK | `agents_layout.sh check-imports --strict-packages` |
| No PyPI `hmem` dependency | OK | Reimplementation under `hmem_core/` |

---

## 2. Paper logic fidelity (intent vs implementation)

| Paper element | Intended behavior | Current code |
|---------------|-------------------|--------------|
| Event extraction | Fine-grained events per fragment (LLM) | **Fixture:** one event = full fragment; regex entity edges (`FixtureEventExtractor`) |
| Temporal windows β_l | Parent window covers children; merge only within window | **Partial** — β partitions + merge; batch `consolidate()` over levels, not incremental insert on each event |
| Semantic threshold α_l | Cluster siblings above α_l | **Partial** — cosine vs threshold; greedy adjacent merge in window |
| Graph entity merge | Disambiguation + profiles | **Partial** — normalize + fuzzy merge; no profile lifecycle |
| Sub-query planning | LLM JSON decomposition + SHORT/LONG/MIXED | **Fixture:** single MIXED sub-query; **LLM** implemented, not CI-tested |
| Sub-query dependencies | Ordered / prerequisite sub-queries | **Not implemented** in `RetrievalEngine` |
| Graph multi-hop | Seed entities → expand → fragment links | **Partial** — hops in `GraphSearcher`; toy graph only |
| Graph seeding | Lexical + vector entity match | **Partial** — text/entity heuristics; no vector entity index |
| Tree bottom-up search | Scope + walk from graph hits | **Partial** — scope filters + fragment allow-list; not full bottom-up walk |
| Ranking F = θ₁S + θ₂T + θ₃R | Full formula | **Implemented** — S/T/R + config weights; keyword overlap nudges S (toy aid) |
| Missing-info follow-up | Re-query when top F is weak | **Stub only** — `missing_info.needs_follow_up` not wired into engine |
| Synthesis Ψ({A_k}) | Per sub-query answers + merge with context | **Single pass:** top-k → one synthesizer call; fixture regex heuristics + LLM path |
| Persistence | Durable tree/graph | **In-memory only** (`hybrid_store.py`) |
| Embeddings in CI | Strong dense model for clustering | **Tests:** `HashDenseEncoder` — weak semantics by design |

**Conclusion:** End-to-end **plan → graph → tree → rank → synthesize** runs without gold-answer shortcut when `runtime.fixture_answer` is unset. Tree consolidation is **heuristic** (community-style greedy merge), not a guaranteed match to incremental parent updates in the paper. **LLM mode** requires `OPENAI_API_KEY` and `HMEM_AGENTS=llm`.

---

## 3. Known omissions (checklist for future passes)

Use this when reopening paper fidelity; each item should have a test or eval slice before merge.

- [ ] LLM event extraction as default ingest; multi-event leaves.
- [ ] Incremental tree insert aligned with paper or community `core/tree.py` + golden merge cases.
- [ ] Entity profiles and richer merge policy in `KnowledgeGraph`.
- [ ] `FixturePlanner` replaced or supplemented so SHORT/LONG/MIXED change evidence in pytest.
- [ ] Wire `needs_follow_up` into `RetrievalEngine` (second retrieval round).
- [ ] Sub-query dependency graph in planner + engine loop.
- [ ] Vector entity seeding for graph search.
- [ ] Remove keyword overlap on S and narrow `FixtureSynthesizer` once encoder/planner are strong.
- [ ] Serialize/load hybrid index under `config.memory_data_dir`.
- [ ] Phase 4: `eval/loaders/`, metrics, optional compare to community harness **ideas** (no vendored code).
- [ ] LLM integration tests (env gate or VCR/mocks).
- [ ] Refresh [`HMEM_INDEX.md`](HMEM_INDEX.md) “Implemented vs planned” table (still partly scaffold-era wording).

**Suggested reopen order:** (1) LoCoMo-mini loader + F1, (2) LLM planner/extractor behind gated test, (3) tree merge goldens, (4) missing-info + multi sub-query synthesis.

---

## 4. API / ingest wiring

| Route | Behavior |
|-------|----------|
| `POST /corpus/fragments` | `HybridIndexer.index_fragment` via `build_indexer` |
| `POST /corpus/bootstrap-toy` | Toy JSON → index + optional fixture QA metadata |
| `POST /index` | `run_consolidation_pass` + `refresh_engine` |
| `POST /ask` | `RetrievalEngine.ask`; gold fallback only if answer empty and fixture metadata set |

---

## 5. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest agents/hmem/tests -q
```

Five tests: health, toy bootstrap+ask, indexer leaves/consolidation, hybrid retrieval without fixture answer.

---

## 6. Remaining work (pointers)

Detailed phase checklist: [`PLAN.md`](PLAN.md). Upstream / no author repo: [`UPSTREAM.md`](UPSTREAM.md).

1. LoCoMo / LongMemEval-style loaders and metrics (Phase 4).
2. LLM integration tests behind env gate or recorded mocks.
3. Tighter paper fidelity per §3 checklist.
4. Shared platform checklist item 2 (Langfuse/Ollama smoke) if desired for parity with siblings.
