# HTML cheatsheet pattern

Open any sheet in a browser (file URL or static server):

```bash
xdg-open research/cheatsheets/index.html
# or: cd research/cheatsheets && python -m http.server 8765
```

## Files

| Path | Role |
|------|------|
| `css/cheatsheet.css` | Layout, green theme, dark mode, `--sheet-read-width` (75rem two-col), `--sheet-read-width-single` (42rem when all cells are `.cell-span`/`.cell-band`), typography |
| `js/cheatsheet.js` | Theme toggle, font scale (0.85–1.25), `localStorage` prefs |
| `js/cheatsheet-nav.js` | Grouped side nav (Databases, Transformers, …) — single source of truth |
| `*.html` | One topic per page; copy structure from `index.html` |

## Page skeleton

1. `.sheet` — grid: fixed nav + sticky header + main
2. `.sheet-nav` — `<div class="nav-groups" data-cheatsheet-nav></div>` + per-page `.nav-footer`
3. `.sheet-header` — `.sheet-title` (sticky while scrolling) + toolbar
4. `.sheet-grid` — two columns by default; auto-narrows to one column (~42rem) if there are no half-width `.cell` sections (see `cheatsheet.js` `applySheetLayout`)
5. `.cell` — padded content; `.cell-title` first line bold
6. **Prose:** expand acronyms on first use in that page; prefer full concise sentences over label-only bullets where the reader may be new to the topic
7. Optional `.cell-diagram` with inline SVG (uses CSS variables)
8. Function shape plots: `<div class="func-chart" data-func-chart="gelu"></div>` — see `js/func-chart.js`
9. **Math:** KaTeX via CDN + `js/katex-render.js`; use `$$…$$` in `.formula` divs and `$…$` in legend / `.sym` (see `loss-functions.html`)

## New page checklist

- [ ] Copy `index.html`, change `<title>`, `.sheet-title`, `.nav-footer`
- [ ] Fill `.sheet-grid` with `<section class="cell">` blocks
- [ ] Add entry under the right category in `js/cheatsheet-nav.js`
- [ ] Scripts: `cheatsheet-nav.js` then `cheatsheet.js` (active link set by nav script)
