# qlab

This is my personal collection of mini projects based on what I'm interested in at the moment. It could be a paper I came across that's interesting or just a concept that I want to reimplement on my own terms just to help me learn.

## Contents

Most of the repo lives under two top-level folders:

- **`agents/`** — LLM agent harness reimplementations and shared runtime (Docker, Langfuse, vector memory helpers). **Run guide:** [`agents/README.md`](agents/README.md).
  - **`reharness/`** — [ReHarness](https://arxiv.org/abs/2610.12668): retrieve reflections, run a tool agent, revise memory from qualitative feedback (Qdrant + LangGraph).
  - **`hipporag/`** — [HippoRAG](https://arxiv.org/abs/2405.14831): knowledge-graph indexing + Personalized PageRank retrieval (multi-hop RAG; **testing**).
  - **`memgraphrag/`** — [MemGraphRAG](https://arxiv.org/abs/2606.00610): multi-agent GraphRAG with shared global memory and hierarchical indexing (**testing**).
  - **`agents_utils/`** — Shared libraries used by agent experiments (Qdrant memory, Langfuse tracing, env/logging).

- **`tutorials/`** — Smaller coding exercises and training sandboxes.
  - **`pathologies/`** — HuggingFace / TRL runs with bad hyperparameters so unhealthy training shows up clearly in Weights & Biases.
    - **`bev/`** — Local W&B viewer: ingest offline `run-*.wandb` into SQLite and explore metrics in the browser.
    - **`doc/`** — Notes on metrics, failure-mode signatures, and how the pathology runs are wired.
  - **`unigram/`** — Unigram language-model tokenizer using Viterbi segmentation over learned piece costs.
  - **`vertibi/`** — Minimal Viterbi decoder over a vocabulary cost map (used by `unigram/`).
