# Upstream code references (H-MEM arXiv:2605.15701)

## Official implementation (authors)

As of scaffold date, **no public GitHub or release** from the paper authors (Jiawei Yu, Yixiang Fang — CUHK-Shenzhen; Xilin Liu, Yuchi Ma — Huawei Cloud) was found on:

- [arXiv:2605.15701](https://arxiv.org/abs/2605.15701) abstract/HTML/PDF
- Web search for `H-MEM` + `2605.15701` + `github`

If authors publish code later, add the URL here and update the module mapping below. **Do not vendor** their repo into qlab; mirror behavior in `hmem_core/` and cite the release tag in `PLAN.md`.

## Community reference (non-official)

| Item | Value |
|------|--------|
| Repo | [Ardha-Eco-System/h-mem](https://github.com/Ardha-Eco-System/h-mem) |
| License | MIT (independent community implementation; **not** affiliated with the paper authors) |
| PyPI | `pip install hmem` (same project) |

Use this repo to **skim algorithms and prompts** when implementing qlab phases. qlab must **not** depend on the `hmem` PyPI package at runtime (import wall + monorepo `agents_utils`).

### Module parity (2026-10-25)

qlab now mirrors community layout under `hmem_core/` (`tree/`, `graph/`, `indexing/*`, `retrieval/*`, `agents/*`, `encoding/`). Stubs are documented in each file header; see [`PLAN.md`](PLAN.md) module map.

### Suggested file mapping (community → qlab)

| Community (`src/hmem/…`) | qlab target (planned) |
|--------------------------|------------------------|
| `core/tree.py` | `hmem_core/tree/temporal_semantic_tree.py` |
| `core/graph.py` | `hmem_core/graph/knowledge_graph.py` |
| `indexing/indexer.py`, `tree_builder.py`, `graph_builder.py` | `hmem_core/indexing/` |
| `retrieval/engine.py`, `planner.py`, `tree_search.py`, `graph_search.py` | `hmem_core/retrieval/` |
| `retrieval/synthesizer.py`, `missing_info.py` | `hmem_core/retrieval/` + optional LLM agents |
| `llm/adapter.py` | `hmem_core/agents/` (fixture + LLM via `agents_utils.llm`) |
| `evaluation/harness.py`, `datasets.py` | `eval/` fixtures first; full benchmarks later |
| `types.py`, `config.py` | `hmem_core/models.py`, `hmem_core/config.py` |

### Smoke test upstream (optional local clone)

```bash
git clone https://github.com/Ardha-Eco-System/h-mem.git /tmp/h-mem-reference
cd /tmp/h-mem-reference && python -m tests.smoke_test
```

Keep clones **outside** `qlab` or in a gitignored path; do not commit third-party trees.
