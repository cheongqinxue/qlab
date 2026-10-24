# Agents

LLM **agent harness** experiments: shared Docker runtime, optional self-hosted **Langfuse**, and per-project compose stacks. Each experiment is a subfolder with its own README for design details; this file is the **how to run** index.

## Layout

| Path | What it is |
|------|------------|
| [`reharness/`](reharness/README.md) | ReHarness continual-learning harness (Qdrant reflections, LangGraph agent, Streamlit UI). |
| [`hipporag/`](hipporag/README.md) | HippoRAG **testing** (KG + PPR retrieval, Neo4j, Streamlit; NeurIPS 2024). |
| [`memgraphrag/`](memgraphrag/README.md) | MemGraphRAG **testing** (multi-agent GraphRAG + global memory; KDD 2026). |
| [`cove/`](cove/README.md) | Chain-of-Verification **testing** (four-step self-check; ACL Findings 2024). |
| [`agents_utils/`](agents_utils/) | Shared Python helpers (encoders, graph PPR, Qdrant memory, Langfuse tracing, logging, `.env`). Experiments must not import each other—only `agents_utils`. |
| [`Dockerfile`](Dockerfile) + [`scripts/build-image.sh`](scripts/build-image.sh) | Shared **`agents-runtime:latest`** image (repo `pyproject.toml` deps + this tree). |
| [`scripts/`](scripts/README.md) | Composable **`start-langfuse`**, **`start-ollama`**, etc. |
| [`.env-template`](.env-template) | Committed template; copy to **`.env`** (gitignored) for all experiments. |
| [`docker-compose.langfuse.yml`](docker-compose.langfuse.yml) | Local Langfuse UI + Postgres/ClickHouse/Redis/MinIO. |
| [`docker-compose.yml`](docker-compose.yml) | Alias: `docker compose up` → Langfuse stack only. |

Persistent data defaults under **`agents/.data/`** (Langfuse), **`reharness/.data/`**, **`hipporag/.data/`**, **`memgraphrag/.data/`**, and **`cove/.data/`** per experiment.

## One-time setup

From the **repository root** (needs [uv](https://github.com/astral-sh/uv) and Docker):

```bash
cp agents/.env-template agents/.env
# Edit agents/.env: OPENAI_API_KEY, Langfuse keys, passwords (see template comments)
```

Build the shared runtime image (repeat after `pyproject.toml` / lockfile changes):

```bash
cd agents && ./scripts/build-image.sh
```

## Shared services (optional but recommended)

ReHarness tracing expects Langfuse on the **`agents_shared`** Docker network.

```bash
./agents/scripts/start-langfuse.sh   # idempotent
./agents/scripts/start-ollama.sh     # local LLM for HIPPORAG_OPENIE=llm / MEMGRAPHRAG_AGENTS=llm
# Langfuse UI: http://localhost:3000 (LANGFUSE_UI_PORT)
```

First boot creates the Langfuse project using `LANGFUSE_INIT_*` and the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env`. Set tracing off with `LANGFUSE_TRACING_ENABLED=false` if you skip this stack.

Stop Langfuse:

```bash
cd agents && docker compose -f docker-compose.langfuse.yml down
```

## Experiments

### ReHarness

Paper-style harness: retrieve reflections → tool-using agent → qualitative feedback → revise memory + forward write.

**Docs:** [`reharness/README.md`](reharness/README.md)

**Run (typical):** two terminals from repo root.

```bash
cd agents && ./reharness/run.sh   # Langfuse + Ollama (if down) + agent + Streamlit → :8501
# Or stepwise: ./reharness/run-agent.sh && ./reharness/run-ui.sh
```

Useful flags:

| Command | Effect |
|---------|--------|
| `./reharness/run-agent.sh --rebuild-image` | Rebuild `agents-runtime:latest` then start. |
| `./reharness/run-agent.sh --detach` | Start detached, health-check, exit (no log follow). |
| `./reharness/run-agent.sh stop` | Stop agent + Qdrant. |
| `./reharness/run-ui.sh` | Host UI; talks to `REHARNESS_AGENT_URL` (default `http://localhost:8080`). |

**Fresh reflection bank:** Streamlit sidebar → confirm → **Reset Qdrant bank**, or `curl -X POST http://localhost:8080/memory/reset`. See [`reharness/README.md`](reharness/README.md) for manual wipe of `.data/qdrant`.

**HTTP API (no UI):**

```bash
curl -N -X POST http://localhost:8080/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{"task":"Your task here"}'
# Then POST feedback to /tasks/{task_id}/feedback/stream
```

**Local Python (no Docker):** possible for development if Qdrant and env vars are set; production path in this repo is the compose stack above.

### HippoRAG

Graph + Personalized PageRank RAG for multi-hop retrieval ([arxiv:2405.14831](https://arxiv.org/abs/2405.14831)). Toy corpus + fixture OpenIE by default; Streamlit tester on **8502**.

**Docs:** [`hipporag/README.md`](hipporag/README.md) · [`hipporag/PLAN.md`](hipporag/PLAN.md)

```bash
cd agents && ./hipporag/run.sh          # Langfuse + Ollama (if down) + agent + Streamlit
# Or stepwise: ./scripts/start-langfuse.sh && ./hipporag/run-agent.sh && ./hipporag/run-ui.sh
curl -s http://localhost:8081/health
./hipporag/run-agent.sh stop
```

Uses port **8081** by default so ReHarness can stay on **8080**. UI port **8502** (`HIPPORAG_UI_PORT`). Set `AGENT_NAME=hipporag` in compose when tracing.

### MemGraphRAG

Multi-agent GraphRAG with **shared memory** (ontology / facts / passages) and conflict-aware indexing ([arxiv:2606.00610](https://arxiv.org/abs/2606.00610)). **Testing** — indexing agents (fixture or LLM), corpus/index/retrieve API, PPR retrieval; no Streamlit UI yet.

**Docs:** [`memgraphrag/README.md`](memgraphrag/README.md) · [`memgraphrag/PLAN.md`](memgraphrag/PLAN.md)

```bash
cd agents && ./memgraphrag/run.sh          # Langfuse + Ollama (if down) + agent
# or: ./memgraphrag/run-agent.sh --detach
curl -s http://localhost:8082/health
./memgraphrag/run-agent.sh stop
```

Default port **8082** (`MEMGRAPHRAG_AGENT_PORT`).

### Chain-of-Verification (CoVe)

Self-verification pipeline: baseline answer, verification questions, **isolated** Q/A, final revision ([arxiv:2309.11495](https://arxiv.org/abs/2309.11495)). Fixture or LLM agents; optional Streamlit UI.

**Docs:** [`cove/README.md`](cove/README.md) · [`cove/PLAN.md`](cove/PLAN.md) · [`cove/COVE_PIPELINE.md`](cove/COVE_PIPELINE.md)

```bash
cd agents && ./cove/run.sh
curl -s http://localhost:8083/health
curl -s -X POST http://localhost:8083/eval/bootstrap-toy
./cove/run-agent.sh stop
```

Default port **8083** (`COVE_AGENT_PORT`). UI: `./cove/run-ui.sh` (default **8503**, `COVE_UI_PORT`). LLM: `COVE_AGENTS=llm`.

---

Future harness experiments should add a row here and their own `README.md`, reuse **`build-image.sh`** and **`agents/.env`**, and join **`agents_shared`** when using Langfuse. See skill [`.cursor/skills/agents-demo-experiment/`](../.cursor/skills/agents-demo-experiment/SKILL.md).

## Environment

Single file: **`agents/.env`**. Key groups:

- **LLM:** `OPENAI_API_KEY`, `OPENAI_MODEL`
- **Langfuse client:** `LANGFUSE_*`, `AGENT_NAME` (trace tag per experiment)
- **ReHarness:** `AGENT_PORT`, `REHARNESS_*`, `QDRANT_*`, `EMBEDDING_MODEL`
- **HippoRAG:** `HIPPORAG_AGENT_PORT`, `HIPPORAG_*`
- **MemGraphRAG:** `MEMGRAPHRAG_AGENT_PORT`, `MEMGRAPHRAG_*`
- **CoVe:** `COVE_AGENT_PORT`, `COVE_AGENTS`

Full list: [`.env-template`](.env-template).

## Adding another experiment

1. Subfolder under `agents/` with FastAPI or worker entrypoint.
2. `docker-compose.yml` using `image: ${AGENTS_RUNTIME_IMAGE:-agents-runtime:latest}` and `env_file: ../.env`.
3. Document run/stop in that folder’s README and link it from the **Experiments** section above.
