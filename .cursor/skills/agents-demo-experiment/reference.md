# `agents_utils` and cross-experiment reuse

Paths relative to repository root.

## `agents/agents_utils/`

| Area | Path | Use for |
|------|------|---------|
| Environment | `env.py` — `load_agents_env()` | Load `agents/.env` before app startup (see MemGraphRAG `main.py`) |
| Logging | `logging_setup.py` — `configure_agent_logging()` | Rotating file logs under `AGENT_LOG_DIR` |
| Observability | `observability/langfuse_tracing.py` | Langfuse client + trace helpers when `LANGFUSE_TRACING_ENABLED` |
| Memory contract | `memory/base_memory.py` — `BaseMemory`, `BaseMemoryItem` | Swappable memory banks (ReHarness Qdrant, MemGraph three-layer adapter) |
| Dense embeddings | `memory/base_encoder.py` — **`BaseEncoder`**: sync `encode_*`, async `aencode_*` (thread pool), `encode_*_numpy` | **Canonical** API for new experiments |
| NumPy helpers | `memory/dense_numpy.py` — `hash_dense_vector`, `dense_cosine_similarity_numpy` | Graph / sync retrieval dot products |
| Dense implementations | `hash_dense_encoder.py`, `fixed_dense_encoder.py`, `sentence_transformers_encoder.py`, `langchain_encoder.py` | Implement sync hooks only; async is inherited |
| Vector memory | `memory/langchain_qdrant_memory.py`, `bm25_sparse.py`, `colbert_encoder.py` | ReHarness-style hybrid retrieval |
| Context window | `context/base_context.py`, `context_manager.py`, `token_counter.py` | Token-budgeted context assembly |
| Web search | `websearch/` — see `INDEX.md` | LangSearch-backed search tool pattern |
| Web fetch | `webfetch/` — see `INDEX.md` | URL fetch + safety helpers |

Example configs (not runtime): `memory/qdrant_memory.config.example.json`.

## Sibling experiments (patterns only — no cross-imports)

Discover runnable harnesses (folders with `main.py` under `agents/`):

```bash
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh list
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh check-imports
```

Skim a **closest sibling** for layout and orchestration patterns (fixture factory, FastAPI lifespan, eval fixtures). **Do not import** another experiment’s `*_core`; promote shared logic into **`agents_utils`** (PPR, encoders, env/logging, etc.).

## Shared ops (not Python)

| Asset | Role |
|-------|------|
| `agents/Dockerfile` + `agents/scripts/build-image.sh` | `agents-runtime:latest` |
| `agents/scripts/start-langfuse.sh` | Langfuse + `agents_shared` network |
| `agents/scripts/start-ollama.sh` | Host or Docker Ollama on `127.0.0.1:11434` |
| `agents/docker-compose.langfuse.yml` | Langfuse stack |
| `agents/.env-template` | Committed env documentation |

## Agent / UI ports (avoid collisions)

Port env vars are listed by:

```bash
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh list
```

Pick the next free `*_AGENT_PORT` / `*_UI_PORT` and document in **`agents/.env-template`**.
