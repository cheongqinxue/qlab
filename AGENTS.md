# Agent instructions (qlab repo)

Read this file at the start of work in this repository.

## How to work with the user

1. **Questions** — Investigate (read code, run commands if needed) and answer. **Do not edit files** unless the user explicitly asks you to implement or change something.
2. **Comments and ideas** — Discuss and clarify. Do not treat them as implicit implementation requests.
3. **Plans** — Produce a plan, options, and tradeoffs only. **Do not implement** until the user gives an explicit go-ahead (e.g. “implement this”, “go ahead”, “fix X”).
4. **Acting** — Edit code, run destructive or wide refactors, commit, or push **only** when the user gives a clear instruction to act.

When in doubt, ask before changing the repo.

## Project pointers

- **Agent harness experiments:** `agents/` — [`agents/README.md`](agents/README.md); **ReHarness** `agents/reharness/`; **HippoRAG** `agents/hipporag/`; **MemGraphRAG** `agents/memgraphrag/`; **CoVe** `agents/cove/`. **Import rule:** experiments must not import each other; shared code lives under **`agents/agents_utils/`** only.
- **New demo experiment workflow:** skill [`agents-demo-experiment`](.cursor/skills/agents-demo-experiment/SKILL.md) (paper + reuse audit, fixtures-first, run scripts).
- **Agents coding norms:** skill [`agents-harness-conventions`](.cursor/skills/agents-harness-conventions/SKILL.md) (import wall, encoders, tests, git).
- **Agents Python review:** skill [`agents-python-review`](.cursor/skills/agents-python-review/SKILL.md) → `agents/REPOINDEX.md`, `agents/CODEREVIEW.md`.
- **Pathology training + W&B logging:** `tutorials/pathologies/`
- **BEV (local W&B viewer UI):** `tutorials/pathologies/bev/` — `backend/bev` (Python), `webui/`, `e2e/`, `sdk/`; see `bev/README.md` for build/run; roadmap in `bev/PLAN.md`; API + DB schema in `bev/docs/backend-reference.md`. Periodic carve-out/packaging audit: skill `bev-distribution-audit` (`.cursor/skills/bev-distribution-audit/`).

## W&B reference (build context)

- **Open-source client:** [wandb/wandb](https://github.com/wandb/wandb) (MIT) — training logs to local `run-*.wandb` files.
- **BEV does not require wandb.ai cloud** — it scans a W&B directory, ingests into SQLite, and serves a local Svelte UI.
- **wandb.ai / full hosted UI** — commercial product; separate from the MIT client.
