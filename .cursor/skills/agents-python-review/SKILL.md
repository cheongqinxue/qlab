---
name: agents-python-review
description: >-
  Review Python under qlab agents/ against global coding rules and agents-harness-conventions.
  Produces or updates agents/REPOINDEX.md and agents/CODEREVIEW.md. Use when the user asks for
  an agents code review, python review of the harness, or a periodic agents/ health check.
---

# Python review — `agents/`

Read first: skill [`agents-harness-conventions`](../agents-harness-conventions/SKILL.md) (project-specific rules).

---

## Global rules to enforce

Cursor **skills do not auto-load** `.mdc` rule files. Naming `python-standards` in prose is not enough — you must **read the rule files** in step 0 below and record that in `CODEREVIEW.md`. Glob-based rules (e.g. `python-standards`) are **not** guaranteed to be in the chat context unless those files are open.

Treat every item in **agents-harness-conventions** (import wall, BaseEncoder, test injection, fixture-first, git/docs) as a **project rule**; tag violations as `agents/<topic>/…` (see template below).

---

## Workflow

### 0. Load global standards (mandatory)

**Before** reviewing code, **read** these rule files from disk (user rules + qlab workspace rule). Do not skip this step.

| Rule id | Read this file |
|---------|----------------|
| **clean-code** | `~/.cursor/rules/clean-code.mdc` |
| **security-standards** | `~/.cursor/rules/security-standards.mdc` |
| **python-standards** | `~/.cursor/rules/python-standards.mdc` |
| **python-tests** | `~/.cursor/rules/python-tests.mdc` (all `agents/**/tests/**`) |
| **fastapi-standards** | `~/.cursor/rules/fastapi-standards.mdc` (`agents/**/server/`, route modules) |
| **collaboration** | `qlab/.cursor/rules/collaboration.mdc` (process only) |

In **`agents/CODEREVIEW.md`**, start with a **Global standards checklist** subsection listing each file above as **read** and bullet the sections you applied (e.g. python-standards: type hints, Google docstrings, pytest-only, tenacity, error handling). Any section not reviewed must say **deferred** and why (e.g. scope limit).

Optional tooling (run when available; note pass/fail in the checklist):

```bash
uv run ruff check agents --statistics
uv run ruff format agents --check --diff
```

If `ruff` is missing from the environment, say so in the checklist and rely on manual review against **python-standards.mdc**.

Tag findings from these files as `clean-code/…`, `security-standards/…`, `python-standards/…`, `python-tests/…`, `fastapi-standards/…`.

### 1. Index the tree

Survey **`agents/`** (experiments, `agents_utils`, scripts, shared compose). Document layout in:

**[`agents/REPOINDEX.md`](../../../agents/REPOINDEX.md)**

- If the file **does not exist**, create it with a concise hierarchy (folders, purpose, main entrypoints).
- If it **exists**, update it for structural changes and add a short **“Last review”** note with date and what changed in the index.

Do not index unrelated repo areas (`tutorials/`, etc.) unless the user asked for a full-repo review.

### 2. Scan for harness boundary violations

From repo root, run the layout helper (see conventions skill):

```bash
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh list
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh check-imports
```

Any reported cross-experiment `*_core` import (including from `agents_utils`) is at least **high** severity. Re-run with `--strict-packages` if you need broader top-level package boundaries.

### 3. Review Python code

Review **all `.py` files under `agents/`** against the **loaded** global rules (step 0) and **agents-harness-conventions**.

Prioritize: `*_core/`, `server/`, `main.py`, `agents_utils/`, and `tests/`.

Minimum **python-standards** pass (from the file you read in step 0):

- Type hints on parameters and returns; prefer `X | None` over `Optional[X]`
- Google docstrings on public classes/functions (`Args` / `Returns` where non-obvious)
- Tests: pytest only, `Test*` naming; test modules uniquely named per experiment (e.g. `test_<experiment>_api.py`)
- Error handling: no bare `except:`; avoid broad `except Exception` except at documented boundaries (prefer specific types)
- Retries on external I/O: tenacity with explicit retriable exceptions (flag ad-hoc retry loops)
- Validation: Pydantic for structured payloads where the codebase already uses it

### 4. Write findings

Append or replace content in:

**[`agents/CODEREVIEW.md`](../../../agents/CODEREVIEW.md)**

Use one file section per source file. Multiple violations per file are grouped under the same `#` heading.

**Entry template** (one `##` block per violation):

```markdown
# path/from/repo/root/agents/...

## Line XX: <rule-id>/<short title>
<Correction urgency: xhigh | high | moderate | low>

```python
# relevant snippet — annotate the violation inline if helpful
```

<What is wrong and which rule/convention it breaks.>

<Suggested correction — minimal diff preferred.>
```

**Rule id examples:**

| Id | Meaning |
|----|---------|
| `security-standards/…` | Secrets, `.env`, logging |
| `clean-code/…` | Naming, size, duplication |
| `python-standards/…` | Types, docstrings, ruff-style |
| `agents/import-boundary/…` | Cross-experiment import |
| `agents/encoder/…` | BaseEncoder, runtime test hacks |
| `agents/tests/…` | Fixture injection, pytest layout |
| `collaboration/…` | Process only when relevant |

**Urgency:**

- **xhigh** — silent failure, data leak, or security
- **high** — breaks behavior or import boundary
- **moderate** — maintenance debt
- **low** — stylistic

If **no violations**, write `agents/CODEREVIEW.md` with a dated summary stating clean review and scope (`agents/`).

### 5. Do not auto-fix

This skill is **review-only** unless the user explicitly asks to implement fixes. Do not commit `CODEREVIEW.md` unless the user asks for a commit.

---

## Scope reminders

- Default scope: **`agents/`** only.
- New experiment workflow is **not** this skill — use [`agents-demo-experiment`](../agents-demo-experiment/SKILL.md).
- Day-to-day coding norms: [`agents-harness-conventions`](../agents-harness-conventions/SKILL.md).
