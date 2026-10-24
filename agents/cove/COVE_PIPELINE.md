# CoVe pipeline (qlab)

Design reference for **Chain-of-Verification** in this experiment. Status labels: **Implemented (scaffold)** vs **Planned**.

Paper: [2309.11495](https://arxiv.org/abs/2309.11495).

## Four steps (paper §3)

| Step | Name | Input | Output | qlab |
|------|------|-------|--------|------|
| 1 | Baseline | User query | Draft response | `BaselineDrafter.draft` — **Implemented (fixture)** |
| 2 | Plan | Query + draft | Verification questions | `VerificationPlanner.plan` — **Implemented (fixture)** |
| 3 | Execute | Query + questions ( **not** draft) | Q/A pairs | `VerificationExecutor.execute` — **Implemented (fixture)** |
| 4 | Revise | Query + draft + Q/A | Final response | `FinalReviser.revise` — **Implemented (fixture)** |

**Invariant (guardrail):** Step 3 must not condition on the baseline draft. The orchestrator passes only `query` and `questions` to the executor.

## Execution modes (paper §3.3)

| Mode | Description | qlab |
|------|-------------|------|
| Joint | One prompt for all verification answers | Planned |
| 2-step | Two chained prompts | Planned |
| Factored | One prompt per verification question | **Default** in config; fixture simulates |

Config: `verification_execution_mode` in `config/default.json` (`factored`).

## HTTP (scaffold)

| Route | Effect |
|-------|--------|
| `GET /health` | Liveness + `COVE_AGENTS` mode |
| `POST /verify` | Run full pipeline (`query` or fixture `query_id`) |
| `POST /eval/bootstrap-toy` | Run toy `query_id=mexican_american_war` |

## Persistence

| Artifact | Status |
|----------|--------|
| In-memory last run | **Planned** (`COVE_STORE_RUNS=1`) |
| JSON run log under `.data/runs/` | Planned |

## Future extensions

- Retrieval-augmented verification answers (external corpus)
- Claim index: map draft sentences → verification questions
- Comparison baseline: same query without CoVe (`POST /baseline` only)
