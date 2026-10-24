# `agents/` repository index

High-level map of the qlab agent harness tree. Entrypoints are **`main.py`** per experiment.

**Last review:** 2026-10-24 — full `agents-python-review` with global `.mdc` checklist; API tests renamed per experiment (`test_*_api.py`).

## Shared (not runnable apps)

| Path | Purpose |
|------|---------|
| [`agents_utils/`](agents_utils/) | Shared libraries: encoders, graph PPR, env/logging, Langfuse, context, web tools (partial), Qdrant memory |
| [`scripts/`](scripts/) | `build-image.sh`, Langfuse/Ollama starters, `_lib.sh` |
| [`Dockerfile`](Dockerfile) | `agents-runtime:latest` image |
| [`.env-template`](.env-template) | Committed env documentation (ports, experiment flags) |
| `docker-compose*.yml` | Langfuse stack, Ollama helper |

## Runnable experiments (`main.py`)

| Folder | Core package | Agent port (default) | UI |
|--------|--------------|-------------------|-----|
| [`reharness/`](reharness/) | `harness`, `graph`, `tools`, `reharness_utils` | 8080 (`AGENT_PORT`) | Streamlit 8501 |
| [`hipporag/`](hipporag/) | `hipporag_core` | 8081 | Streamlit 8502 |
| [`memgraphrag/`](memgraphrag/) | `memgraphrag_core` | 8082 | — |
| [`cove/`](cove/) | `cove_core` | 8083 | Streamlit 8503 |

Typical layout per experiment:

```
<name>/
├── main.py, config/default.json, docker-compose.yml
├── <name>_core/     # domain logic
├── server/          # FastAPI routes + app_state + settings
├── tests/           # pytest (unique module names e.g. test_<experiment>_api.py)
├── eval/            # toy fixtures (where applicable)
├── run-agent.sh, run.sh, _run_common.sh
└── ui/              # Streamlit (optional)
```

## Import rule

Experiments must not import each other's `*_core`. Shared code belongs in **`agents_utils/`**.

Check: `bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh check-imports`

## Related docs

- Run index: [`README.md`](README.md)
- Skills: `.cursor/skills/agents-harness-conventions/`, `agents-python-review/`, `agents-demo-experiment/`
