# HippoRAG implementation plan (qlab)

**Primary spec:** NeurIPS 2024 paper [2405.14831](https://arxiv.org/abs/2405.14831) (PDF in this folder: `2405.14831v3.pdf`).

**Official code:** [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) (cloned for review; not vendored into qlab).

| Git ref | What it is |
|---------|------------|
| **`main`** (pip `hipporag` 2.x) | **HippoRAG 2** — [2502.14802](https://arxiv.org/abs/2502.14802) (ICML ’25). Fact reranking, passage nodes in-graph, DSPy filter, vector-store backends, MCP. |
| **`legacy` / tag `v1.0.0`** | **HippoRAG 1** era — single `src/hipporag.py`, closer to the NeurIPS paper experiments. |
| **`reproduce/` + `main.py`** | Batch eval driver (MuSiQue / 2Wiki / Hotpot JSON under `reproduce/dataset/`). |

**qlab goal:** A **FastAPI agent** under `agents/hipporag/` that reimplements the **HippoRAG 1 retrieval loop** (OpenIE index → query linking → PPR → ranked passages → optional QA), aligned with the paper and cross-checked against **`legacy`** and (where useful) **`main`** for prompts and hyperparameters. We do **not** ship the full upstream package or reproduce benchmark leaderboard numbers in v1.

---

## Paper pipeline (what to build)

### Offline indexing (memory encoding)

1. **Passages** `P` — chunks of the corpus (qlab: user-provided text + stable ids).
2. **OpenIE (LLM `L`, temp 0)** — per passage, **two steps** (Appendix I / upstream `prompts/templates/ner.py` + `triple_extraction.py`):
   - Extract **named entities** (JSON list).
   - Extract **OpenIE triples** `(subject, predicate, object)` using those entities as hints (noun-phrase nodes, not only NER).
3. **Graph index** — nodes `N` from triple phrases; edges `E` from triple structure; optional **synonymy edges** `E′` when `cos(M(n_i), M(n_j)) ≥ τ` (paper: **τ = 0.8**).
4. **Passage–node matrix** — paper defines **`P` ∈ ℝ^{|N|×|P|}** (counts of each node in each passage). Use this or an equivalent sparse mapping for final passage ranking.

**Pattern separation:** discrete phrases + graph edges (not one vector per passage only).

### Online retrieval (pattern completion)

1. **Query entities `C_q`** — LLM NER on the question (`ner_query.py` in upstream).
2. **Query nodes `R_q`** — map each query entity to a graph node via **argmax cosine** with encoder **`M`** (same as indexing).
3. **Personalization vector** — uniform mass on `R_q`, zero elsewhere; apply **node specificity** `s_i = 1 / |P_i|` (passages that mention node `i`) to query-node weights before PPR (paper §2.3).
4. **PPR** on undirected KG (`E ∪ E′`), damping **0.5** (paper default; tuned on MuSiQue train in paper).
5. **Passage scores** — **`p = n′ᵀ P`** (paper); rank passages, return top‑k.
6. **Optional (paper Appendix F.2):** if any query→node similarity **< θ**, **ensemble** normalized HippoRAG scores with dense passage retrieval (Contriever-style).

### QA (downstream)

Retrieve top‑k passages → single LLM reader (not the core retrieval contribution). Upstream: `qa_top_k=5`, `max_qa_steps=1` for single-step RAG.

---

## How upstream differs (read before coding)

Use this so qlab’s plan stays **paper-faithful** without accidentally building HippoRAG 2.

| Topic | Paper / legacy v1 | Upstream `main` (v2) |
|--------|-------------------|----------------------|
| PPR library | **igraph** `personalized_pagerank` | Same (`run_ppr`, `directed=False`, `prpack`) |
| Graph | Entity + triple edges + synonymy; passage scores via **matrices** | **Passage/chunk nodes in graph**, **fact nodes**, weighted fact+synonym edges |
| Query → graph | NER + entity link + PPR seeds | **Fact retrieval** + **DSPy rerank** → `graph_search_with_fact_entities` |
| Dense retrieval | Optional uncertainty ensemble | **Always** mixes DPR into PPR via `passage_node_weight` (default **0.05**) |
| Embeddings | Contriever / ColBERTv2 in paper | Default **`nvidia/NV-Embed-v2`**; optional Qdrant/Chroma/Milvus stores |
| OpenIE | 1-shot NER → triples | Same flow in `information_extraction/openie_openai.py` |
| Synonymy | Similarity threshold **0.8** | **KNN** (`synonymy_edge_topk`, batch sizes) + threshold **0.8** |
| Damping | **0.5** in paper | **`damping=0.5`** in `BaseConfig` (legacy script default was **0.1** — prefer paper/config for qlab) |

**Recommendation:** Implement **Phase 1–3** from the **paper + legacy** mental model; add **v2-style passage-node DPR seeding** and fact reranking only in **Phase 5** if needed for parity with `main`.

---

## Graph storage: Neo4j (chosen)

**Why Neo4j:** the shipped **Neo4j Browser** (and optional Bloom) makes the hippocampal index **visible**—you can confirm OpenIE triples, synonymy links, and bridge entities (e.g. Prof. Thomas between “Stanford” and “Alzheimer’s”) without building a custom graph UI. Same spirit as the paper; different engine than upstream’s igraph pickle.

**Tradeoff:** one more Docker service and the **Graph Data Science (GDS)** library for **Personalized PageRank** in-process (or export subgraph → igraph for PPR in Python if we want to avoid GDS licensing/plugin setup on day one).

### Property graph schema (v1)

| Label | Key properties | Notes |
|-------|----------------|--------|
| `Passage` | `id`, `text`, `content_hash` | Source chunks |
| `Entity` | `id` (hash), `text` (normalized phrase) | OpenIE noun phrases / entities |
| `REL` | `predicate`, `passage_id`, `weight` | Triple: `(Entity)-[:REL]->(Entity)` |
| `SYNONYM` | `score` | Undirected synonymy when cos ≥ τ |
| `MENTIONS` | `count` | `(Entity)-[:MENTIONS]->(Passage)` for node specificity & passage ranking |

Indexing uses **`MERGE`** on `Entity` / `Passage` so re-runs are idempotent. OpenIE raw JSON still cached under **`.data/graph/openie/`** (cheap re-parse without LLM).

### Browser visualization (operator workflow)

After `POST /index`, open **http://localhost:7474** (compose port from env) and run queries such as:

```cypher
// Overview
MATCH (p:Passage) RETURN count(p) AS passages;
MATCH (e:Entity) RETURN count(e) AS entities;
MATCH ()-[r:REL]->() RETURN count(r) AS triple_edges;

// One passage and its extractions
MATCH (p:Passage {id: $passage_id})<-[:MENTIONS]-(e:Entity)
OPTIONAL MATCH (e)-[r:REL]->(e2:Entity)
RETURN p, e, r, e2;

// Bridge hunt (multi-hop intuition): entities linked to 2+ passages
MATCH (e:Entity)-[:MENTIONS]->(p:Passage)
WITH e, collect(p.id) AS ps WHERE size(ps) >= 2
RETURN e.text, ps LIMIT 25;
```

For PPR debugging, temporarily **highlight seed entities** returned in `/retrieve` debug JSON (same ids as `Entity.id`).

### PPR on Neo4j

1. **Preferred (in DB):** GDS **`gds.pageRank.stream`** (or mutate) with **`sourceNodes`** = query-linked entities, **`dampingFactor`** = 0.5, on a **projected** undirected graph (`REL` + `SYNONYM`, relationship weights). Map ranked **`Entity`** scores → **`Passage`** via `MENTIONS` (paper: aggregate node mass onto passages; equivalent to **n′ᵀ P** when `MENTIONS.count` encodes **P**).

2. **Fallback (simpler bootstrap):** Cypher **`MATCH`** export of adjacency + **`igraph`** PPR in Python (reference implementation parity), while **all writes** still go to Neo4j for Browser viz.

Phrase **embeddings** for synonymy and query linking stay in Python (`SentenceTransformersEncoder` or small parquet cache)—Neo4j holds structure, not vectors, unless we later add vector indexes.

---

## qlab architecture (target)

```
agents/hipporag/
  main.py                 # FastAPI lifespan: Neo4j driver + Index + Retriever
  hipporag_core/
    openie.py             # LLM NER + triple extraction (structured JSON)
    encoder.py            # Phrase embeddings for link + synonymy
    neo4j_store.py        # MERGE passages/entities/edges; Cypher helpers
    graph_store.py          # Facade over neo4j_store (+ optional igraph PPR export)
    synonymy.py             # KNN + τ → MERGE SYNONYM
    indexer.py              # index_passages()
    retriever.py            # query NER → link → PPR → passage ranks
    ppr.py                  # GDS PageRank or igraph fallback
    prompts.py              # OpenIE prompt shapes (upstream templates)
  server/routes/
    corpus.py               # POST passages, POST index, POST reset
    retrieve.py             # POST /retrieve
    ask.py                    # POST /ask
  eval/toy_multihop/
  config/default.json
  docker-compose.yml        # agent + neo4j (+ optional neo4j volume)
  .data/hipporag/neo4j/     # Neo4j DB files (gitignored)
  .data/graph/openie/       # OpenIE JSON cache (gitignored)
```

**Reuse from qlab:** `agents_utils` (env, logging, Langfuse, embeddings), shared Docker image, `ChatOpenAI`.

**New deps (when implementing):** `neo4j` (Python driver). **GDS** runs in the **Neo4j container** (image/plugin choice in compose), not necessarily in pip. Optional: `igraph` only if using PPR fallback.

---

## Phases

### Phase 0 — Scaffold ✅

- [x] Experiment folder, Docker (**8081**), health API, README, local PDF
- [ ] Add **`eval/toy_multihop/`** — 5–10 passages, 3–5 questions (Stanford / Alzheimer’s / bridge entity); expected supporting passage ids

### Phase 1 — Indexing (vertical slice)

- [ ] **`POST /corpus/passages`** — append `{id, text}`; idempotent re-index skips unchanged hashes
- [ ] Port **OpenIE prompts** from upstream `src/hipporag/prompts/templates/ner.py` and `triple_extraction.py` (keep 1-shot examples; wire to `OPENAI_MODEL`)
- [x] **Neo4j in compose** — service on `hipporag_internal` network; Browser on host port **7474** (HTTP) / **7687** (Bolt); credentials in `agents/.env`
- [x] **GDS Community plugin** — `NEO4J_PLUGINS='["graph-data-science"]'` on `neo4j:5.26-community`; jars under `.data/hipporag/neo4j/plugins/`
- [ ] **`POST /index`** — OpenIE → **`MERGE`** into Neo4j (`Passage`, `Entity`, `REL`, `MENTIONS`)
- [ ] **`add_synonymy_edges`** — embedding KNN + **`MERGE` `SYNONYM`** (τ = 0.8; all-pairs OK on toy corpus)
- [ ] OpenIE JSON cache under **`.data/graph/openie/`**
- [ ] **Done when:** `GET /health` reports counts; **Browser** shows triples for a indexed passage (Cypher above)

### Phase 2 — Retrieval (paper-faithful)

- [ ] Query NER prompt from upstream `ner_query.py`
- [ ] Link query entities → graph nodes (argmax cosine with **`M`**)
- [ ] Build reset weights: uniform on linked nodes × **node specificity**
- [ ] **PPR** — GDS personalized PageRank from seed `Entity` nodes (**damping=0.5**) or igraph fallback on exported adjacency
- [ ] Passage ranking — aggregate entity scores onto `Passage` via **`MENTIONS`** (same spirit as **n′ᵀ P**)
- [ ] **`POST /retrieve`** — `{query, top_k}` → passages + scores + debug (seed nodes, top nodes)
- [ ] **Done when:** toy question *“Which Stanford professor works on the neuroscience of Alzheimer’s?”* ranks the **bridge passage** above single-hop distractors; dense-only baseline fails

### Phase 3 — QA agent

- [ ] **`POST /ask`** — retrieve `qa_top_k` (default 5) → RAG prompt → answer
- [ ] Langfuse spans: `openie`, `index`, `retrieve`, `ppr`, `generate` (`AGENT_NAME=hipporag`)
- [ ] Optional **LangGraph** mirror of ReHarness: linear `retrieve → generate` (no reflection bank)
- [ ] **Done when:** `/ask` correct on toy eval JSON

### Phase 4 — Evaluation & baselines

- [ ] Script: recall@2 / @5 on `eval/toy_multihop/`
- [ ] **Dense baseline** — same encoder, no graph (upstream `StandardRAG` / `retrieve_dpr` behavior)
- [ ] Optional: load upstream **`reproduce/dataset/sample.json`** + **`sample_corpus.json`** for a slightly larger smoke test (no full MuSiQue download)
- [ ] Document results in `README.md` (not CI gates initially)

### Phase 5 — Optional parity / hardening

- [ ] Uncertainty ensemble (θ) when min query–node similarity is low (paper F.2)
- [ ] **Passage nodes + DPR seed** (`passage_node_weight`) like upstream `graph_search_with_fact_entities`
- [ ] **Fact scoring + rerank** (HippoRAG 2 path) — only if deliberately targeting `main` behavior
- [ ] Incremental index / delete passage (upstream `index()` incremental paths)
- [x] Streamlit UI (`run-ui.sh`) — bootstrap toy, retrieve, ask, reset

---

## Hyperparameters (defaults for qlab)

| Parameter | Value | Source |
|-----------|-------|--------|
| `synonymy_edge_sim_threshold` / τ | **0.8** | Paper §3.4, upstream `BaseConfig` |
| `damping` | **0.5** | Paper §3.4, upstream `BaseConfig` |
| `retrieval_top_k` / `qa_top_k` | **5** (API), **200** (internal retrieve then trim) | Paper metrics use R@2/R@5; upstream `retrieval_top_k=200` |
| `linking_top_k` | **5** | Upstream (v2 fact linking; optional in v1) |
| `passage_node_weight` | **0.05** | Upstream v2 only — Phase 5 |
| LLM temperature | **0** | Paper |
| Encoder | **`facebook/contriever`** or qlab `EMBEDDING_MODEL` | Paper; document swap if scores drop |

---

## Reference map (official repo)

When implementing, prefer reading these upstream paths (on **`main`**, unless noted):

| Concern | Path |
|---------|------|
| End-to-end class | `src/hipporag/HippoRAG.py` — `index()`, `retrieve()`, `run_ppr()`, `add_synonymy_edges()` |
| Config defaults | `src/hipporag/utils/config_utils.py` — `BaseConfig` |
| OpenIE online | `src/hipporag/information_extraction/openie_openai.py` |
| Prompts | `src/hipporag/prompts/templates/ner.py`, `triple_extraction.py`, `ner_query.py` |
| KNN synonymy | `src/hipporag/utils/embed_utils.py` — `retrieve_knn` |
| Dense baseline | `src/hipporag/StandardRAG.py` |
| v1 monolith | **`legacy:src/hipporag.py`** — PPR + `node_specificity` + matrix doc scores |
| Sample data | `reproduce/dataset/sample.json`, `sample_corpus.json` |

---

## Non-goals

- Full MuSiQue / 2Wiki / Hotpot reproduction or claiming SOTA
- Vendoring or pip-depending on **`hipporag`** inside qlab (reference only)
- HippoRAG 2 feature-complete port before paper v1 toy eval passes
- IRCoT multi-step loop (optional later; paper compares to IRCoT as baseline)

---

## Acceptance summary

| Milestone | Criterion |
|-----------|-----------|
| **M1** | Phase 1 + 2 on toy corpus; pytest on PPR ranking |
| **M2** | `/ask` + Langfuse on toy eval |
| **M3** | Dense vs HippoRAG recall@k documented on toy + optional sample dataset |

After **M1**, the project is no longer “empty scaffold” — reasonable to commit as **`feat(hipporag): indexing and PPR retrieval`**.
