---
name: agents-harness-conventions
description: >-
  qlab agents/ harness conventions: import boundaries, BaseEncoder wiring, fixture-first tests,
  collaboration and git norms. Apply when editing or reviewing code under agents/, adding shared
  utilities, or refactoring any runnable harness under agents/. Read before implementing
  cross-cutting agent changes.
---

# `agents/` harness conventions

Read first: [`AGENTS.md`](../../../AGENTS.md), [`agents/README.md`](../../../agents/README.md).

For **scaffolding a new experiment**, use skill [`agents-demo-experiment`](../agents-demo-experiment/SKILL.md) instead of duplicating that checklist here.

---

## Global rules (apply in this tree)

Use your **global Cursor rules**; the ones that matter under `agents/` are:

| Rule id | Scope |
|---------|--------|
| **collaboration** | Question vs plan vs act; no edits/commits unless the user asks |
| **clean-code** | Readable names, small functions, minimal branching |
| **security-standards** | `.env` gitignored, `.env-template` committed, no secrets in code/logs |

Also follow the user’s **Python / testing** preferences when writing or reviewing code here: ruff (100 cols), type hints, Google docstrings on public APIs, pytest (`Test*` classes, `@pytest.mark.asyncio` for async), minimal diffs, no over-engineering.

---

## Import boundary (non-negotiable)

**Experiments must not import each other.**

| Allowed | Forbidden |
|---------|-----------|
| `agents_utils.*` | Another experiment’s `<name>_core` from outside that folder |
| Same experiment’s `*_core`, `server`, local `main` | Any cross-experiment `*_core` import |
| Third-party packages | Putting another experiment on `sys.path` to reuse its code |

Shared algorithms and infra live only under **`agents/agents_utils/`** (e.g. `memory/base_encoder.py`, `graph/ppr.py`, env, logging, Langfuse helpers, web tools).

When two harnesses need the same logic, **move it to `agents_utils`**, then import from both sides.

Discover experiments and verify **`*_core`** import boundaries (from repo root):

```bash
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh list
bash .cursor/skills/agents-harness-conventions/scripts/agents_layout.sh check-imports
```

`list` prints each harness (folder with `main.py`), its top-level packages, port hints from `agents/.env-template`, and experiments that share `tests/test_api.py`. Use `--strict-packages` on `check-imports` to include non-`*_core` top-level packages (e.g. ReHarness `tools/`).

---

## Encoders and runtime wiring

- **Canonical type:** `agents_utils.memory.base_encoder.BaseEncoder` (`encode_document` / `encode_query`, numpy helpers).
- **Production:** `build_runtime(..., encoder=None)` resolves to a real encoder (e.g. `SentenceTransformersEncoder` via `EMBEDDING_MODEL`)—not test doubles.
- **Tests:** inject via **pytest fixtures + `monkeypatch`** on `build_runtime` (or direct constructor args). Use `FixedDenseEncoder` / `HashDenseEncoder` for deterministic vectors.
- **Avoid:** `PYTEST_CURRENT_TEST`, `*_TEST_ENCODER`, or other pytest detection **inside runtime/production paths**.
- **Avoid:** unnecessary `create_*` encoder factories when `encoder=` injection suffices.

MemGraph-style paths that need vectors should **require** an encoder when indexing/retrieving—no silent hash default that hides misconfiguration.

---

## Tests and API smoke

- **Fixture-first:** toy `eval/` JSON, fixture agents, bootstrap env flags before LLM paths.
- **API tests:** load each experiment’s `main.py` with **`importlib`** (`spec_from_file_location`) so `main` / `server` do not collide across harnesses.
- **Pytest invocation:** run **one experiment per invocation** when possible:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest agents/<name>/tests -q
```

Do not combine multiple experiment test dirs in one pytest run when `list` reports a shared basename (typically `test_api.py`); use `agents/<name>/tests` per harness.

---

## Git and docs

- **Commits / push** only when the user explicitly asks.
- Prefer focused commits: `feat: …`, `chore: document …`, `chore: update setup and run scripts`.
- Never commit `agents/.env` or credentials.
- When boundaries or shared modules change, update **`AGENTS.md`**, **`agents/README.md`**, and the experiment’s **`PLAN.md` / `MEMORY_GRAPH.md`** as needed—not unrelated markdown.

---

## Related skills

| Skill | Use when |
|-------|----------|
| [`agents-demo-experiment`](../agents-demo-experiment/SKILL.md) | Adding a new paper harness |
| [`agents-python-review`](../agents-python-review/SKILL.md) | Full Python review under `agents/` with written findings |

Shared utilities inventory: [`agents-demo-experiment/reference.md`](../agents-demo-experiment/reference.md).
