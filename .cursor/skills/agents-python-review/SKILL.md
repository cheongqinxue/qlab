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

Apply your **global Cursor rules** to all Python under **`agents/`**:

- **clean-code**
- **security-standards**
- **collaboration** (only relevant if the review task includes changing files without explicit user ask—note as process, not a line violation)

Also apply the user’s **Python / testing** standards (ruff, types, docstrings, pytest patterns) as **python-standards**-class findings when violated.

Treat every item in **agents-harness-conventions** (import wall, BaseEncoder, test injection, fixture-first, git/docs) as a **project rule**; tag violations as `agents/<topic>/…` (see template below).

---

## Workflow

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

Review **all `.py` files under `agents/`** against the global rules and **agents-harness-conventions**.

Prioritize: `*_core/`, `server/`, `main.py`, `agents_utils/`, and `tests/`.

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
