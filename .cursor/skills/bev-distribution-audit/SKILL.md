---
name: bev-distribution-audit
description: >-
  Audits BEV (tutorials/pathologies/bev) for carve-out, pip install, and Docker readiness.
  Checks sandbox boundaries, thin SDK, packaging guardrails, and PLAN alignment. Use when the
  user asks for a BEV distribution audit, carve-out check, packaging readiness review, or
  periodic BEV architecture health check before Phase 4–5.
---

# BEV distribution & carve-out audit

Read `tutorials/pathologies/bev/PLAN.md` (repository home, Phase 4–5, guardrails) first.

**Product intent (do not drift):**

- BEV moves to **its own repo**; **qlab** is a personal sandbox only.
- Default UX: **one process** — API + bundled static UI (`bev serve` / `bev-backend start`).
- **`bev_sdk` is a thin client:** callback/training scaffolding + HTTP to the backend (e.g. ingest
  nudge). No W&B parsing, SQLite, or ingest pipeline logic in the SDK.

## When to run

- Before large SDK/server work, before carve-out, or when the user requests a periodic check.
- After refactors that touch `backend/`, `sdk/`, `webui/`, CLI, or paths.

## Audit workflow

1. **Scan boundaries** (run from repo root):

```bash
# No imports from pathology / qlab tutorials into BEV packages
rg -n "from pathologies|import pathologies|tutorials/pathologies" \
  tutorials/pathologies/bev/backend tutorials/pathologies/bev/sdk --glob '*.py' || true

# Runtime hard-codes to qlab repo layout (review each hit)
rg -n "tutorials/pathologies/bev|PYTHONPATH.*tutorials" \
  tutorials/pathologies/bev/backend/bev tutorials/pathologies/bev/sdk --glob '*.py' || true
```

2. **Static / packaging paths**

- `bev.config.static_dir()` / `product_root()`: OK for **monorepo dev**; installed wheel must use
  `importlib.resources` (flag if production serve path still only uses `product_root()`).
- `static/` should remain gitignored; release must build `webui` in CI, not commit assets.

3. **SDK thin-client**

- Grep `bev_sdk` and `sdk/` for: `wandb`, `sqlite`, `ingest`, `store`, `pipeline` — should not
  appear except re-exports/docs/tests that mock HTTP.
- Ingest nudge API (when present) should live in `backend/bev/server/`; SDK only calls it.

4. **API / UI split**

- `/api/*` JSON vs `/` static — preserve for single-host and reverse proxy.
- If UI talks to another origin: require documented `BEV_API_URL` + CORS (flag if split deploy
  assumed without config).

5. **Trust model** (note current state vs PLAN)

- Local default: `127.0.0.1`, no auth — OK for dev.
- Flag if nudge API is added without bind-address or optional shared-secret story for Docker/LAN.

6. **Tests still runnable in sandbox**

```bash
uv run pytest tutorials/pathologies/bev/backend/tests tutorials/pathologies/bev/sdk/tests -q --tb=no
```

(Optional) `cd tutorials/pathologies/bev/e2e && npm test` if UI/API touched.

7. **PLAN.md freshness**

- Open decisions table matches reality (own repo, pip shape, static resources).
- Phase 4 carve-out checklist still accurate.

## Report template

Deliver this structure (no file edits unless the user asked to fix findings):

```markdown
# BEV distribution audit — <date>

## Summary
<1–2 sentences: green / yellow / red>

## Critical (blocks carve-out or pip)
- ...

## Warnings (fix before Phase 4)
- ...

## OK / unchanged
- ...

## Suggested next actions
1. ...
```

Severity:

- **Critical:** `pathologies` imports in BEV, ingest logic in SDK, committed `static/`, secrets in repo.
- **Warning:** new repo-relative paths in runtime code, missing nudge API contract doc, CORS gap for split UI.
- **OK:** scripts-only PYTHONPATH, docs mentioning sandbox paths, dev-only `product_root()`.

## Do not

- Treat pathology `run.sh` / `.data` as ownership of BEV source.
- Recommend merging BEV into the qlab `pyproject.toml` for PyPI.
- Expand SDK scope without user approval (keep thin client).

## Reference

- Plan: `tutorials/pathologies/bev/PLAN.md`
- API/schema: `tutorials/pathologies/bev/docs/backend-reference.md`
- Agent pointer: `AGENTS.md` (BEV section)
