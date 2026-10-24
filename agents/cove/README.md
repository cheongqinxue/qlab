# Chain-of-Verification (CoVe) experiment

qlab harness for **Chain-of-Verification** — reduce hallucinations by drafting, planning verification questions, answering them in **isolation**, then revising ([arxiv:2309.11495](https://arxiv.org/abs/2309.11495)).

## Paper

**Chain-of-Verification Reduces Hallucination in Large Language Models** — Dhuliawala et al., ACL Findings 2024.

| | |
|--|--|
| PDF in repo | [`papers/2309.11495.pdf`](papers/2309.11495.pdf) |
| Abstract | [arxiv.org/abs/2309.11495](https://arxiv.org/abs/2309.11495) |

**Idea:** (1) baseline response → (2) verification questions for factual claims → (3) answer each question **without** the draft in context → (4) final verified response.

## Comparison in qlab

| | **ReHarness** | **CoVe (this folder)** |
|--|---------------|------------------------|
| Goal | Continual harness / reflections | **Single-query** self-verification |
| Memory | Qdrant bank | Stateless runs (session log planned) |

| | **HippoRAG / MemGraphRAG** | **CoVe** |
|--|----------------------------|----------|
| Retrieval | Corpus graph / memory | Closed-book LLM steps (RAG optional later) |

Shared stack: **`agents-runtime:latest`**, **`agents/.env`**, optional Langfuse (`AGENT_NAME=cove`).

## Status

**Testing** — four-step orchestrator, fixture agents (`COVE_AGENTS=fixture`) and LLM agents (`COVE_AGENTS=llm`), `POST /verify`, toy eval, Streamlit UI (`run-ui.sh`).

See [`PLAN.md`](PLAN.md) and [`COVE_PIPELINE.md`](COVE_PIPELINE.md).

### Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest agents/cove/tests -q
```

### Run

```bash
cd agents && ./cove/run.sh
curl -s http://localhost:8083/health
curl -s -X POST http://localhost:8083/verify -H 'Content-Type: application/json' \
  -d '{"query_id":"mexican_american_war"}'
# UI (separate terminal): ./cove/run-ui.sh  → http://127.0.0.1:8503
```

LLM mode (requires `OPENAI_API_KEY` or Ollama via `OPENAI_BASE_URL`):

```bash
COVE_AGENTS=llm ./cove/run-agent.sh --detach
curl -s -X POST http://localhost:8083/verify -H 'Content-Type: application/json' \
  -d '{"query":"When did the Mexican–American War start and end?"}'
```
