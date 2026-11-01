<p align="center">
  <img src="assets/logo.svg" alt="qlab logo" width="220">
</p>

# qlab

This is my personal collection of mini projects based on what I'm interested in at the moment. It could be a paper I came across that's interesting or just a concept that I want to reimplement on my own terms just to help me learn.

## Cheatsheets

Static HTML reference pages under [`research/cheatsheets/`](research/cheatsheets/). Open in a browser via local HTTP (from that folder: `python -m http.server`); GitHub file preview does not run the JS bundles.

- **[Local model eval matrix](research/cheatsheets/agents-model-eval.html)** — Ollama quant comparison from [`agents/model_eval/`](agents/model_eval/) (`export_cheatsheet_data.py` to refresh).
- **[Cheatsheet index](research/cheatsheets/index.html)** — all tracks (training, agents, MLOps, …).

## Contents

Most of the repo lives under two top-level folders:

- **`agents/`** — LLM agent harness reimplementations and shared runtime (Docker, Langfuse, vector memory helpers). **Run guide:** [`agents/README.md`](agents/README.md).
  - **`reharness/`** — [ReHarness](https://arxiv.org/abs/2610.12668) targets **agentic continual learning**: agents reuse past textual “reflections,” but retrieved lessons can be wrong or overgeneral for new tasks. The method wraps the decision agent in a **recurrent harness** that retrieves relevant reflections before each instance, writes a new one afterward, and **revises only the reflections that were retrieved**, using the latest trajectory and feedback so memory stays grounded and correctable over time.
  - **`hipporag/`** — [HippoRAG](https://arxiv.org/abs/2405.14831) addresses **multi-hop open-domain QA**, where dense RAG misses connections that require linking many pieces of background knowledge. It builds a **lightweight entity–relation graph** from open information extraction, then retrieves via **Personalized PageRank** seeded by query entities—an indexing-and-retrieval scheme inspired by hippocampal indexing theory rather than brute-force passage similarity alone.
  - **`memgraphrag/`** — [MemGraphRAG](https://arxiv.org/abs/2606.00610) tackles **GraphRAG at scale**, where per-chunk extraction yields **fragmented, conflicting, and inconsistent** graphs. It introduces **shared global memory** (ontology, facts, passages) maintained by a **multi-agent** pipeline that extracts, detects conflicts, and resolves them with corpus-wide context, then indexes a **hierarchical graph** for structure-aware, memory-guided retrieval.
  - **`cove/`** — [Chain-of-Verification](https://arxiv.org/abs/2309.11495) mitigates **LLM hallucination** on factual questions where a single greedy answer hides unsupported claims. **CoVe** drafts a response, generates **verification questions** for its claims, answers those questions **in isolation** (without the draft in context), and **revises** the answer from the verified evidence—reducing confirmation bias in self-checking.
  - **`hmem/`** — [H-MEM](https://arxiv.org/abs/2605.15701) studies **long-horizon agent memory**: storing and retrieving experience over time without flat logs or purely vector stores. It couples a **temporal–semantic tree** (short-term events consolidated into long-term summaries under similarity and time windows) with an **entity knowledge graph** for multi-hop lookup, plus a hybrid **online retrieval** ranker over semantic, temporal, and robustness cues.
  - **`agents_utils/`** — Shared libraries used by agent experiments (Qdrant memory, Langfuse tracing, env/logging).

- **`tutorials/`** — Smaller coding exercises and training sandboxes.
  - **`pathologies/`** — HuggingFace / TRL runs with bad hyperparameters so unhealthy training shows up clearly in Weights & Biases.
    - **`bev/`** — Local W&B viewer: ingest offline `run-*.wandb` into SQLite and explore metrics in the browser.
    - **`doc/`** — Notes on metrics, failure-mode signatures, and how the pathology runs are wired.
  - **`unigram/`** — Unigram language-model tokenizer using Viterbi segmentation over learned piece costs.
  - **`vertibi/`** — Minimal Viterbi decoder over a vocabulary cost map (used by `unigram/`).

## Development

Install git hooks (Ruff lint + format, basic file hygiene):

```bash
uv sync --group dev
pre-commit install
pre-commit run --all-files   # optional: check entire tree before first commit
```

Lint rules live in `pyproject.toml` under `[tool.ruff.lint]` (includes blind/bare `except` checks via `BLE` / `E722`).
