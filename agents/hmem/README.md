# H-MEM (qlab harness)

Reimplementation of **H-MEM: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure** ([arXiv:2605.15701](https://arxiv.org/abs/2605.15701); PDF in [`papers/2605.15701v1.pdf`](papers/2605.15701v1.pdf)).

Hybrid **temporal–semantic tree** (short-term events → long-term summaries) plus **entity knowledge graph** for multi-hop memory QA.

**Status:** **testing** — hybrid index + retrieval pipeline, toy pytest green; LoCoMo-scale eval and exact paper merge semantics in [`PLAN.md`](PLAN.md) Phase 4 / fidelity checklist.

**Design:** [`HMEM_INDEX.md`](HMEM_INDEX.md) · **Review:** [`IMPLEMENTATION_REVIEW.md`](IMPLEMENTATION_REVIEW.md)

## Comparison (qlab siblings)

| Harness | Memory index | Evolution | Multi-hop |
|---------|--------------|-----------|-----------|
| ReHarness | Qdrant vectors | Reflection rewrite | No |
| HippoRAG | KG + PPR | Static corpus index | Yes (PPR) |
| MemGraphRAG | Global graph layers | Conflict-aware merge | Yes (graph + PPR) |
| **H-MEM** | Tree + graph + vectors | Temporal consolidation | Yes (KG traversal) |

## Upstream code

See [`UPSTREAM.md`](UPSTREAM.md) — **no author repo found**; community reference [Ardha-Eco-System/h-mem](https://github.com/Ardha-Eco-System/h-mem) for algorithm skim only.

## Run (when stack is wired)

```bash
cd agents && ./hmem/run-agent.sh
curl -s http://localhost:8084/health | jq
```

Details: [`PLAN.md`](PLAN.md), [`agents/README.md`](../README.md).

## Citation

```bibtex
@article{yu2026hmem,
  title={H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure},
  author={Yu, Jiawei and Fang, Yixiang and Liu, Xilin and Ma, Yuchi},
  journal={arXiv preprint arXiv:2605.15701},
  year={2026}
}
```
