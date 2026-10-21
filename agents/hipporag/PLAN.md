# HippoRAG implementation plan (qlab)

Reference: [2405.14831](https://arxiv.org/abs/2405.14831), official [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG).

## Phase 0 — Scaffold (current)

- [x] Experiment folder, Docker compose, health API, docs
- [ ] Pin retrieval eval corpus (e.g. small JSON passages + multi-hop questions)

## Phase 1 — Offline indexing

- [ ] Passage ingest API (`POST /corpus/passages`)
- [ ] LLM OpenIE: subject–predicate–object triples per passage (reuse `ChatOpenAI` from shared stack)
- [ ] Graph store: nodes (entities, passages), edges (triple links + passage–entity mentions)
- [ ] Persist graph under `hipporag/.data/graph/` (NetworkX or SQLite adjacency — decide in Phase 1)

## Phase 2 — Online retrieval

- [ ] Query → extract seed entities / facts (LLM or NER-lite)
- [ ] Build PPR reset vector over graph nodes (entity + passage weights from paper)
- [ ] Run PPR (`networkx` or `igraph`); rank passages by score
- [ ] `POST /retrieve` → top-k passage IDs + optional LLM answer endpoint

## Phase 3 — Agent integration

- [ ] Optional LangGraph: retrieve → generate (compare to ReHarness reflection retrieve)
- [ ] Langfuse spans for index vs retrieve vs generate
- [ ] Streamlit or minimal CLI for demo queries

## Phase 4 — Evaluation

- [ ] One multi-hop mini-benchmark in-repo (5–10 questions, 20–50 passages)
- [ ] Baselines: dense passage RAG vs HippoRAG single-step

## Non-goals (for now)

- Full reproduction of paper MuSiQue / Hotpot numbers
- Vendoring the entire official HippoRAG repo (use as reference only)
