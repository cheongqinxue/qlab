# MemGraphRAG experiment

qlab scaffold for **MemGraphRAG** — a memory-based **multi-agent** GraphRAG system with shared global memory and hierarchical graph indexing, alongside [`../reharness/`](../reharness/README.md) and [`../hipporag/`](../hipporag/README.md).

## Paper

**MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation**

Chuanjie Wu, Zhishang Xiang, Yunbo Tang, Zerui Chen, Qinggang Zhang, Jinsong Su — **KDD 2026**.

| | |
|--|--|
| Abstract + PDF | [arxiv.org/abs/2606.00610](https://arxiv.org/abs/2606.00610) |
| HTML | [arxiv.org/html/2606.00610](https://arxiv.org/html/2606.00610) |
| Official code | [github.com/XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) |

**Idea (one paragraph):** Fragment-level GraphRAG extraction yields **inconsistent, conflicting, disconnected** graphs on large corpora. MemGraphRAG uses **shared global memory** (ontology schemas, extracted facts, source passages) and a **society of agents** to extract, detect conflicts, and resolve them with corpus-wide context. The **hierarchical indexing graph** built from memory supports **memory-aware retrieval** (filtering, structure-aware seeds, PPR-style ranking).

## Comparison in qlab

| | **HippoRAG** | **MemGraphRAG (this folder)** |
|--|--------------|-------------------------------|
| Graph build | Per-passage OpenIE + PPR | Multi-agent extraction + **conflict resolution** via shared memory |
| Memory tiers | Passage + entity graph | **Ontology / fact / passage** layers |
| Agents | Mostly pipeline LLM calls | Explicit **multi-agent** roles |

| | **ReHarness** | **MemGraphRAG** |
|--|---------------|-----------------|
| Goal | Continual **harness** reflections | **Corpus** GraphRAG QA |

Shared stack: **`agents-runtime:latest`**, **`agents/.env`**, optional Langfuse (`AGENT_NAME=memgraphrag`).

## Status

**Scaffold only** — health API and folder layout. See [`PLAN.md`](PLAN.md).

## Quick start

```bash
cd agents && ./build-image.sh
./memgraphrag/run-agent.sh
curl -s http://localhost:8082/health
./memgraphrag/run-agent.sh stop
```

Default port **8082** (`MEMGRAPHRAG_AGENT_PORT`).

## Layout

| Path | Role |
|------|------|
| [`memgraphrag_core/`](memgraphrag_core/) | Global memory tiers, agent stubs, retrieval (TBD) |
| [`server/`](server/) | FastAPI |
| [`config/default.json`](config/default.json) | Defaults |
| [`PLAN.md`](PLAN.md) | Implementation phases |
