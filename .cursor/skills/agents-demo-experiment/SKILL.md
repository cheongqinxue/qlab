---
name: agents-demo-experiment
description: >-
  Repeatable workflow to add a new qlab agents/ harness experiment (paper reimplementation,
  local demo, fixture-first tests). Covers paper and upstream code review, agents_utils reuse audit,
  experiment layout, eval fixtures, Docker run scripts, and status docs. Use when the user asks to
  scaffold or build a new agent under agents/, start a demo experiment (e.g. Chain-of-Verification),
  or follow the same pattern as ReHarness, HippoRAG, or MemGraphRAG.
---

# New `agents/` demo experiment

Read first: [`AGENTS.md`](../../../AGENTS.md) (collaboration: plan vs implement), [`agents/README.md`](../../../agents/README.md) (run index).

Copy this checklist into the experiment **`PLAN.md`** and tick steps as you go:

```markdown
## Delivery checklist
- [ ] 1 Sources (paper PDF, upstream repo skim, README + PLAN Phase 0)
- [ ] 2 Shared platform (.env, build-image, optional Langfuse/Ollama)
- [ ] 3 Reuse audit (agents_utils + sibling *_core — documented in PLAN)
- [ ] 4 Experiment core (<name>_core, server, config, tests layout)
- [ ] 5 Fixture path (eval/, factory, bootstrap, pytest)
- [ ] 6 LLM path (optional; env flag, Ollama via host.docker.internal)
- [ ] 7 Runnable stack (compose, _run_common, run-agent, run.sh)
- [ ] 8 Scale/design doc (MEMORY_GRAPH.md or equivalent, if graphs or heavy persistence)
- [ ] 9 Register docs (root README, agents/README, status scaffold → testing)
- [ ] 10 Commits (feat / chore docs / chore scripts — user-requested only)
```

---

## 1. Sources first

- **Paper:** If no PDF in-repo, add under `agents/<name>/papers/` (or document a stable URL in README). Filename should include arXiv id or venue year when known.
- **`PLAN.md` Phase 0 (mandatory before large code):**
  - Read the paper: data flow, modules, evaluation protocol.
  - Find **official code**; clone or browse and note files to mirror vs simplify.
  - Map paper concepts → qlab folders (memory bank? graph? multi-agent roles? harness loop?).
- **`README.md`:** citation, one-paragraph idea, comparison table vs ReHarness / HippoRAG / MemGraphRAG.
- **Register** folder in root [`README.md`](../../../README.md) and [`agents/README.md`](../../../agents/README.md) (status **scaffold** until runnable).

---

## 2. Shared platform

One-time or after dependency changes (from repo root):

```bash
cp agents/.env-template agents/.env   # edit locally; never commit secrets
cd agents && ./scripts/build-image.sh # → agents-runtime:latest
```

Optional session services:

```bash
./agents/scripts/start-langfuse.sh
./agents/scripts/start-ollama.sh
```

- Missing **`agents-runtime:latest`:** `run-agent.sh` / `run.sh` **warn and exit**; build via `scripts/build-image.sh` or **`--rebuild-image`**.
- LLM from agent containers: `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`, compose **`extra_hosts: host.docker.internal:host-gateway`**.

Details: [`agents/scripts/README.md`](../../../agents/scripts/README.md).

---

## 3. Reuse audit (before writing experiment core)

**Goal:** Same qlab patterns as existing experiments; extend shared code instead of duplicating.

1. Read [`reference.md`](reference.md) — **`agents_utils`** and cross-experiment reuse map.
2. List current harnesses and pick a **closest sibling** to skim (`main.py` lifespan, `server/app_state.py`, routes, `config/default.json`, `_run_common.sh`):

```bash
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh list
```

3. Write a **Reuse** subsection in **`PLAN.md`**: chosen base classes, what stays experiment-specific. **Do not** import other experiments’ `*_core` — promote shared code to `agents_utils` (run `check-imports` after changes).

**Prefer:**

| Need | First look |
|------|------------|
| Memory bank contract | `agents_utils.memory.base_memory.BaseMemory` |
| Embeddings / encoders | **`agents_utils.memory.base_encoder.BaseEncoder`** + `SentenceTransformersEncoder` / `LangchainEncoder` / `HashDenseEncoder` (tests) |
| Context / token budget | `agents_utils.context` |
| Env + logging + Langfuse | `agents_utils.env`, `logging_setup`, `observability.langfuse_tracing` |
| Web tools | `agents_utils.websearch`, `webfetch` |
| PPR (RAG graphs) | `agents_utils.graph.ppr` |

Do **not** assume every paper uses `BaseMemory` — the audit step decides.

---

## 4. Experiment code shape (domain core)

Under `agents/<name>/`:

```
<name>/
├── README.md, PLAN.md
├── config/default.json
├── docker-compose.yml
├── main.py
├── _run_common.sh, run-agent.sh, run.sh
├── <name>_core/          # paper-specific logic
├── server/               # FastAPI routes + app_state
└── tests/
```

- Implement **paper-specific** orchestration, models, and pipelines on top of the reuse audit.
- Use **`typing.Protocol`** (or ABC when shared base behavior is real) for roles that will have **fixture vs LLM** implementations later.
- **`AGENT_NAME=<name>`** for Langfuse; assign a unique **`*_AGENT_PORT`** — use `agents_layout.sh list` (reads `agents/.env-template`) and pick the next free port.

---

## 5. Fixture path and demo facilities (separate from core)

Build **before** LLM agents:

- **`eval/`** — toy corpus + JSON fixtures for intermediate outputs (conflicts, extractions, verification steps, etc.).
- **Fixture role implementations** + **`factory`** + env (`<NAME>_AGENTS=fixture|llm` or project convention).
- **Bootstrap:** `MEMGRAPHRAG_BOOTSTRAP_TOY`-style env and/or `POST /corpus/bootstrap-toy` (adapt path names to the paper).
- **Tests:** fixture pipeline + API smoke; run per experiment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest agents/<name>/tests -q
```

Avoid importing two experiments’ `main`/`server` in one pytest invocation.

---

## 6. LLM path (after fixtures pass)

- LLM role classes, prompts, structured outputs; wire factory when `*_AGENTS=llm`.
- Document Ollama / OpenAI env vars in **`agents/.env-template`** (placeholders only).

---

## 7. Runnable stack

Mirror existing experiments:

- **`docker-compose.yml`:** agent service, `agents_shared` network, volumes under `<name>/.data/`.
- **`_run_common.sh`:** `ensure_agents_runtime_image`, `ensure_agents_shared_network` (may call `start-langfuse.sh`).
- **`run.sh`:** optional Langfuse → Ollama → `run-agent.sh --detach` → Streamlit UI if applicable (`--no-langfuse`, `--no-ollama`, `--agent-only`).

---

## 8. Design doc for future scale

When the paper uses graphs, Neo4j, or non-trivial retrieval:

- Add **`MEMORY_GRAPH.md`** (or named equivalent): authoritative vs derived state, update semantics, PPR/retrieval path, **Implemented (testing)** vs **Planned**.
- Cross-link an existing graph harness **`MEMORY_GRAPH.md`** (if any under `agents/*/MEMORY_GRAPH.md`) when reusing PPR/Neo4j patterns.

---

## 9. Status and documentation

- Experiment README **Status:** **scaffold** → **testing** when fixture demo + API + tests + `run.sh` work (UI optional).
- Update root and **`agents/README.md`** layout table and run section.
- Extend **`.env-template`** with new ports/flags only.

---

## 10. Git (only when user asks)

Typical split:

1. `feat: built <name>` — core, tests, eval, compose for agent
2. `chore: document …` — MEMORY_GRAPH / design references
3. `chore: update setup and run scripts` — if touching shared `agents/scripts/`
4. `chore: update readme for <name> status`

Never commit `agents/.env` or secrets.

---

## Reference

- Shared utilities inventory: [reference.md](reference.md)
- Collaboration rules: [`.cursor/rules/collaboration.mdc`](../../rules/collaboration.mdc)
