# HippoRAG experiment

qlab reimplementation scaffold for **HippoRAG** — hippocampal-indexing-style RAG with a knowledge graph and Personalized PageRank (PPR), sitting alongside [`../reharness/`](../reharness/README.md) (reflection harness / continual learning).

## Paper

**HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su — **NeurIPS 2024**.

| | |
|--|--|
| Abstract + PDF | [arxiv.org/abs/2405.14831](https://arxiv.org/abs/2405.14831) |
| HTML | [arxiv.org/html/2405.14831](https://arxiv.org/html/2405.14831) |
| Official code | [github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |

**Idea (one paragraph):** Dense passage RAG encodes each chunk in isolation, so **multi-hop** questions that require linking facts across passages are weak. HippoRAG mimics **hippocampal indexing**: an LLM extracts **OpenIE-style triples** from passages into a **knowledge graph** (neocortex-like storage), then at query time seeds **Personalized PageRank** on that graph to pull jointly relevant passages in **one retrieval step**—often matching iterative RAG at lower cost.

## vs ReHarness in this repo

| | **ReHarness** | **HippoRAG (this folder)** |
|--|----------------|----------------------------|
| Memory shape | Textual reflections in Qdrant | Passage + entity graph |
| Update loop | Forward write + backward revision from feedback | Offline index new passages; online PPR retrieve |
| Typical use | Agent harness / style & task lessons | Multi-hop QA over a corpus |

Both can share **`agents-runtime:latest`**, **`agents/.env`**, and optional Langfuse (`AGENT_NAME=hipporag`).

## Status

**Scaffold only** — health API and layout; indexing + PPR not wired yet. See [`PLAN.md`](PLAN.md).

## Quick start

```bash
cd agents && ./build-image.sh   # if image missing or deps changed
./hipporag/run-agent.sh
curl -s http://localhost:8081/health | jq
./hipporag/run-agent.sh stop
```

Default port **8081** (`HIPPORAG_AGENT_PORT`) so ReHarness can keep **8080**.

## Layout

| Path | Role |
|------|------|
| [`hipporag_core/`](hipporag_core/) | Indexing, graph store, PPR retrieval (stubs → implementation) |
| [`server/`](server/) | FastAPI routes and settings |
| [`config/default.json`](config/default.json) | Graph / PPR defaults |
| [`PLAN.md`](PLAN.md) | Implementation phases |
