# CoVe implementation plan (qlab)

Reference: [2309.11495](https://arxiv.org/abs/2309.11495), PDF in [`papers/2309.11495.pdf`](papers/2309.11495.pdf).

## Delivery checklist

- [x] 1 Sources (paper PDF, README + PLAN Phase 0)
- [ ] 2 Shared platform (.env, build-image, optional Langfuse/Ollama)
- [x] 3 Reuse audit (documented below)
- [x] 4 Experiment core (`cove_core/`, `server/`, config)
- [x] 5 Fixture path (`eval/`, factory, bootstrap API, pytest)
- [ ] 6 LLM path (`COVE_AGENTS=llm`, factored execution prompts)
- [x] 7 Runnable stack (compose, `_run_common.sh`, `run-agent.sh`, `run.sh`)
- [x] 8 Pipeline design doc ([`COVE_PIPELINE.md`](COVE_PIPELINE.md))
- [x] 9 Register docs (root + `agents/README.md`, status **scaffold**)
- [ ] 10 Commits (user-requested)

## Phase 0 — Sources

- [x] Paper PDF in `papers/`
- [x] Read four-step CoVe loop (baseline → plan → execute isolated → revise)
- [x] Note execution variants: joint, 2-step, **factored** (independent prompt per question — default for qlab)
- [ ] Skim few-shot prompts in paper appendix for LLM phase

**Official code:** none required; community references (e.g. Instructor CoVe doc) inform API shape only.

**qlab mapping:**

| Paper step | Module |
|------------|--------|
| Baseline | `agents/baseline.py`, `BaselineDrafter` protocol |
| Plan verifications | `agents/plan.py`, `VerificationPlanner` |
| Execute (factored) | `agents/execute.py`, `VerificationExecutor` — **no draft in context** |
| Final verified | `agents/revise.py`, `FinalReviser` |
| Orchestration | `orchestrator.py` |

No `BaseMemory` for v1 — runs are stateless; optional run history JSON under `.data/runs/` later.

## Reuse audit

| Source | Used for |
|--------|----------|
| `agents_utils.env` | `load_agents_env` in `main.py` |
| `agents_utils.logging_setup` | Agent file logs |
| `agents_utils.observability.langfuse_tracing` | Optional traces per CoVe step |
| `langchain_openai.ChatOpenAI` | Planned LLM phase (same as MemGraphRAG) |
| MemGraphRAG patterns | FastAPI layout, `_run_common.sh`, fixture factory, pytest path purge |

**Not used:** `BaseMemory`, HippoRAG PPR, Neo4j, Qdrant.

**Encoders:** When CoVe needs embeddings (RAG verification later), use **`agents_utils.memory.base_encoder.BaseEncoder`** (e.g. `SentenceTransformersEncoder`; tests inject `HashDenseEncoder`).

## Phase 1 — Scaffold (current)

- [x] `CoVeOrchestrator` + fixture quartet
- [x] `POST /verify`, `POST /eval/bootstrap-toy`, `GET /health`
- [x] Toy fixture: Mexican–American War dates (draft with error → revision)

## Phase 2 — LLM agents

- [ ] `LlmBaselineDrafter`, planner, factored executor, reviser
- [ ] Structured outputs / prompts aligned with paper appendix
- [ ] Langfuse spans per step

## Phase 3 — Demo UX

- [ ] Streamlit UI showing four stages side-by-side
- [ ] Optional retrieval-augmented verification (paper §3.3)

## Phase 4 — Eval

- [ ] MultiSpanQA / list QA fixtures
- [ ] Metrics: claim-level accuracy vs baseline
