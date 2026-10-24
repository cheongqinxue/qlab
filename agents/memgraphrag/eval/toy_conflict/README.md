# Toy conflict corpus (fixture index)

Minimal eval set for **hard conflict** demo (e.g. conflicting Newton birth-year facts across passages).

| File | Role |
|------|------|
| `corpus.json` | Passage ids + text for M_pas staging |
| `extraction.json` | Per-passage A_ext fixture (schemas + facts) |
| `conflicts.json` | A_det fixture conflict groups |
| `resolutions.json` | A_res fixture drop/keep actions by target fact id |
| `memory.json` | Optional pre-resolved snapshot for bootstrap |

Set `MEMGRAPHRAG_EXTRACTION=fixture` (future env) and bootstrap like HippoRAG toy load.
