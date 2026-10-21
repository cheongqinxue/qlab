# Pathology lab documentation (draft)

This folder is for **human-facing explanations** of the training-pathology tutorial: what each metric means, how to read W&B panels, failure-mode signatures, and how the code is wired. The runnable tutorial stays in the parent directory (`train.py`, `run.sh`, tests).

## How GitHub displays content (important)

| Format | Browsing the repo on github.com | Real “website” (nav, search, theme) |
|--------|----------------------------------|-------------------------------------|
| **Markdown** (`.md`) | Rendered in the file view | Needs **GitHub Pages** (or similar) |
| **HTML** (`.html`) | Shown as **source**, not a live page | Works on **GitHub Pages** when deployed |
| **Jupyter** (`.ipynb`) | Rendered if outputs are saved in the notebook | Same; or export to HTML for Pages |

So: dropping HTML in `doc/` does **not** make GitHub show a rendered site by itself. You either read Markdown/ notebooks in the repo UI, or you **publish** static files (usually via GitHub Pages).

## Options (short)

### 1. Markdown only in `doc/` (lowest friction)

- Write `doc/metrics.md`, `doc/failure-modes.md`, link from `pathologies/README.md`.
- **Pros:** No build step; looks good on GitHub; easy to review in PRs.
- **Cons:** No single site with sidebar/search; feels like a folder of articles, not a manual.

### 2. GitHub Pages + **MkDocs Material** (recommended for a real site)

- Source: Markdown under `doc/` (or repo-root `docs/`).
- Build: `mkdocs build` → static HTML; GitHub Actions deploys to Pages.
- **Pros:** Proper site (nav, search, dark mode, admonitions, mermaid); still Markdown in git; matches “explain everything” goal.
- **Cons:** Small CI/config footprint (`mkdocs.yml`, workflow).

### 3. **Jupyter notebooks** in `doc/notebooks/` or `notebooks/`

- **Pros:** Great for “run and see” plots, interactive exploration, narrative + code.
- **Cons:** Heavy diffs; outputs must be committed for GitHub preview; poor fit for reference tables (metric glossary, W&B panel map); not a substitute for a structured manual.

**Hybrid that works well:** MkDocs for the manual + optional notebooks linked as “labs” (notebooks render on GitHub; the book links to them).

### 4. Hand-written HTML in `doc/`

- **Pros:** Full control.
- **Cons:** Pain to maintain; no advantage over MkDocs unless you need custom JS. Still requires Pages to feel like a website.

## Suggested layout (if we adopt MkDocs later)

```text
doc/
  README.md                 ← this file
  index.md                  ← landing / overview
  getting-started.md        ← run.sh, W&B local server, env
  metrics/                  ← glossary + what to watch in W&B
  failure-modes/            ← one page per mode + chart signatures
  instrumentation/          ← hooks, hook_targets.yaml, telemetry presets
  operations/               ← workspace setup, cleanup scripts, batch runs
```

Generated artifacts (e.g. `hook_targets.yaml`) can be **linked** or **included** via MkDocs `pymdown` snippets—not duplicated by hand.

## GitHub Pages wiring (when you choose)

Typical for a monorepo tutorial:

1. Repo **Settings → Pages → Build from GitHub Actions**.
2. Workflow runs on `main`, builds from `tutorials/pathologies/doc` with `mkdocs.yml` beside it or at `tutorials/pathologies/mkdocs.yml`.
3. Site URL: `https://<user>.github.io/<repo>/` (or project subpath if using `site_url`).

Alternative: publish only the pathology section as a subpath—needs `site_url` / `docs_dir` configured once.

## What to decide next

1. **Scope:** Pathology tutorial only (`tutorials/pathologies/doc/`) vs whole `qlab` repo?
2. **Primary format:** Markdown manual (MkDocs) vs notebooks-first vs repo Markdown only?
3. **Hosting:** GitHub Pages on this repo, or private/local only for now?

Reply with those three and we can scaffold `mkdocs.yml` + first pages, or start with plain Markdown chapters only.

## Failure-mode notes (started)

Research-backed pages (detection + cited fixes) live under **[failure-modes/](failure-modes/README.md)** — one Markdown file per pathology / best-practice mode.
