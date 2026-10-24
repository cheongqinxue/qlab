# Agents scripts

Composable shell steps for local testing. **Experiment-specific** compose and UI stay under `hipporag/`, `memgraphrag/`, etc.

## Why scripts + compose (not one mega-compose)

| Layer | Tool | Reason |
|-------|------|--------|
| Langfuse, Ollama | `docker-compose.*.yml` + `start-*.sh` | Heavy, shared, optional; start once per dev session. |
| Agent + Neo4j / Qdrant | per-experiment `docker-compose.yml` | Different ports, volumes, env; you often run **one** agent at a time. |
| Streamlit UI | host `uv run streamlit` | Fast iteration; no need to rebuild the runtime image. |

A single monolithic compose file would restart Langfuse/Ollama every time you touch HippoRAG, or force all experiments up together. **Scripts orchestrate; compose defines services.**

## Scripts

| Script | Role |
|--------|------|
| [`build-image.sh`](build-image.sh) | Build `agents-runtime:latest` (occasional). |
| [`start-langfuse.sh`](start-langfuse.sh) | Langfuse stack + `agents_shared` network (idempotent). |
| [`stop-langfuse.sh`](stop-langfuse.sh) | Tear down Langfuse compose. |
| [`start-ollama.sh`](start-ollama.sh) | Ensure Ollama API on `127.0.0.1:11434` (host or Docker). |
| [`stop-ollama.sh`](stop-ollama.sh) | Stop **Docker** Ollama only. |

Shared helpers: [`_lib.sh`](_lib.sh) (paths, env, health waits).

## Agent entrypoints

| Experiment | One-shot test |
|------------|----------------|
| ReHarness | [`../reharness/run.sh`](../reharness/run.sh) |
| HippoRAG | [`../hipporag/run.sh`](../hipporag/run.sh) |
| MemGraphRAG | [`../memgraphrag/run.sh`](../memgraphrag/run.sh) (agent only; no UI yet) |

If `agents-runtime:latest` is missing, `run-agent.sh` / `run.sh` exit with a message pointing at **`scripts/build-image.sh`**. Use **`--rebuild-image`** on `run-agent.sh` or `run.sh` to build then start.

Flags on `run.sh`: `--no-langfuse`, `--no-ollama`, `--agent-only` (skip Streamlit where applicable).

## LLM from Docker agents → host Ollama

In `agents/.env`:

```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_MODEL=qwen2.5:7b-instruct
```

Experiment compose files add `extra_hosts: ["host.docker.internal:host-gateway"]` on the agent service so this resolves on Linux.
