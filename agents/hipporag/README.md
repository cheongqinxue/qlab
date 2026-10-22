# HippoRAG experiment

qlab reimplementation of **HippoRAG** — hippocampal-indexing-style RAG with a knowledge graph and Personalized PageRank (PPR), sitting alongside [`../reharness/`](../reharness/README.md) (reflection harness / continual learning).

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

**Testing** — M1 vertical slice (indexing + PPR retrieval): Neo4j + fixture toy corpus, igraph PPR, FastAPI + Streamlit UI, pytest on toy multi-hop. Not production-hardened (see [`PLAN.md`](PLAN.md) for remaining phases). Paper PDF: `2405.14831v3.pdf`.

### Tests

From repo root (no Neo4j / no HF download in pytest):

```bash
uv run python -m pytest agents/hipporag/tests -q
```

**Note:** Upstream **`main`** is **HippoRAG 2**; qlab targets **HippoRAG 1** ([2405.14831](https://arxiv.org/abs/2405.14831)) first — see PLAN for `legacy` vs `main` differences.

## Quick start

```bash
cd agents && ./build-image.sh   # if image missing or deps changed
./hipporag/run-agent.sh
curl -s http://localhost:8081/health | jq
# Graph is pre-loaded: eval/toy_multihop (fixture OpenIE → Neo4j). Browser: http://localhost:7474
curl -s -X POST http://localhost:8081/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which Stanford professor works on Alzheimer'\''s?","top_k":3}' | jq
./hipporag/run-agent.sh stop
./hipporag/run-ui.sh   # Streamlit on http://127.0.0.1:8502 (agent must be up)
```

On startup the agent loads **`eval/toy_multihop/corpus.json`** and indexes with **`openie.json`** (zero OpenIE LLM calls). Re-run manually: `POST /corpus/bootstrap-toy`. Live OpenIE: set **`HIPPORAG_OPENIE=llm`** and **`HIPPORAG_BOOTSTRAP_TOY=0`** in `agents/.env`.

Default port **8081** (`HIPPORAG_AGENT_PORT`) so ReHarness can keep **8080**.

Neo4j ships **GDS Community** (free, all algorithms; 4-core compute cap) via `NEO4J_PLUGINS` in compose. Plugin JARs persist under `hipporag/.data/hipporag/neo4j/plugins/`. After first start, in Browser run `RETURN gds.version();` — retrieval still uses **igraph PPR** in Python until we wire GDS as an optional backend.

## Layout

| Path | Role |
|------|------|
| [`hipporag_core/`](hipporag_core/) | OpenIE, memory + **Neo4j** graph, igraph PPR, indexer / retriever |
| [`tests/`](tests/) | Pytest unit + API tests (toy corpus fixtures) |
| [`eval/toy_multihop/`](eval/toy_multihop/) | Mini corpus, OpenIE fixtures, eval questions |
| [`server/`](server/) | FastAPI routes and settings |
| [`config/default.json`](config/default.json) | Graph / PPR defaults |
| [`PLAN.md`](PLAN.md) | Implementation phases |
