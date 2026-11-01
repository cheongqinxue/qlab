/**
 * Agents model_eval results: sortable matrix + model detail modal.
 */
(function () {
  /** @typedef {{ id: string, label: string }} Dim */
  /** @typedef {{ case_id: string, dimension: string, prompt: string, final_answer: string, passed: boolean, failure_modes: string[], signals?: { infrastructure?: { categories?: string[] } } }} CaseRow */
  /** @typedef {{ id: string, name: string, params?: string, ollama_tag?: string, total_passed: number, total_cases: number, runtime?: { wall_s?: number, case_latency_s?: { mean?: number, sum?: number } }, gpu_vram_mib?: { peak_used_mib?: number, mean_peak_used_mib?: number, source?: string }, metrics_backfill?: Record<string, string>, token_stats?: { completion_tokens?: number, prompt_tokens?: number, total_tokens?: number, completion_tok_per_s?: number, total_tok_per_s?: number, denominator_source?: string, cases_with_usage?: number }, by_dimension: Record<string, { passed: number, total: number }>, cases: CaseRow[] }} ModelRow */

  const data = window.AGENTS_MODEL_EVAL_DATA;
  if (!data) {
    return;
  }

  /** @type {Dim[]} */
  const dimensions = data.dimensions || [];
  /** @type {ModelRow[]} */
  let models = [...(data.models || [])];
  /** @type {{ key: string, dir: 'asc' | 'desc' }} */
  let sortState = { key: "total_pct", dir: "desc" };
  /** @type {string} */
  let dialogActiveView = dimensions[0]?.id || "tool_use";
  /** @type {ModelRow | null} */
  let dialogModel = null;

  const table = document.getElementById("eval-matrix");
  const tbody = document.getElementById("eval-matrix-body");
  const metaEl = document.getElementById("eval-meta");
  const dialog = document.getElementById("eval-detail-dialog");
  const dialogTitle = document.getElementById("eval-detail-title");
  const dialogViewBar = document.getElementById("eval-dialog-view-bar");
  const dialogBody = document.getElementById("eval-detail-body");

  if (!table || !tbody || !dialog || !dialogTitle || !dialogBody) {
    return;
  }

  if (metaEl) {
    metaEl.textContent = `Snapshot ${data.generated_at || "unknown"} — ${data.suite_note || ""}`;
  }

  /**
   * @param {number} passed
   * @param {number} total
   */
  function pct(passed, total) {
    if (!total) {
      return 0;
    }
    return Math.round((1000 * passed) / total) / 10;
  }

  /**
   * @param {ModelRow} model
   * @param {string} dimId
   */
  function dimCell(model, dimId) {
    const block = model.by_dimension[dimId];
    if (!block || !block.total) {
      return { text: "—", sort: -1, title: "No cases" };
    }
    const p = pct(block.passed, block.total);
    return {
      text: `${block.passed}/${block.total}`,
      sort: p,
      title: `${p}% passed`,
    };
  }

  /**
   * @param {ModelRow} model
   */
  function runtimeCell(model) {
    const wall = model.runtime?.wall_s;
    if (wall == null || Number.isNaN(wall)) {
      return { text: "—", sort: -1, title: "No runtime recorded" };
    }
    const sec = Math.round(wall);
    let text;
    if (sec < 60) {
      text = `${sec}s`;
    } else if (sec < 3600) {
      const m = Math.floor(sec / 60);
      const r = sec % 60;
      text = r ? `${m}m ${r}s` : `${m}m`;
    } else {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      text = m ? `${h}h ${m}m` : `${h}h`;
    }
    const mean = model.runtime?.case_latency_s?.mean;
    let title =
      mean != null
        ? `${wall.toFixed(1)} s wall clock · ~${mean.toFixed(1)} s mean per case`
        : `${wall.toFixed(1)} s wall clock (full suite)`;
    if (model.metrics_backfill?.runtime) {
      title += " · backfilled from JSONL latency sum (no harness overhead)";
    }
    const est = Boolean(model.metrics_backfill?.runtime);
    return { text: est ? `${text}†` : text, sort: wall, title, estimated: est };
  }

  /**
   * @param {ModelRow} model
   */
  /**
   * @param {ModelRow} model
   */
  function completionTokCell(model) {
    const stats = model.token_stats;
    const tps = stats?.completion_tok_per_s;
    if (tps == null || Number.isNaN(tps)) {
      return {
        text: "—",
        sort: -1,
        title: "No completion token usage in saved traces (re-run or backfill metrics)",
      };
    }
    const comp = stats.completion_tokens ?? 0;
    const total = stats.total_tokens ?? comp;
    const totalTps = stats.total_tok_per_s;
    const src = stats.denominator_source === "wall_s" ? "suite wall clock" : "sum of case latencies";
    let title = `Out ${tps} tok/s · ${comp.toLocaleString()} completion tokens over ${src}`;
    if (totalTps != null) {
      title += ` · ${totalTps} total tok/s (${total.toLocaleString()} in+out)`;
    }
    const text = tps >= 100 ? String(Math.round(tps)) : tps.toFixed(1);
    return { text, sort: tps, title };
  }

  function vramCell(model) {
    const gpu = model.gpu_vram_mib;
    const mib = gpu?.peak_used_mib;
    if (mib == null || Number.isNaN(mib)) {
      return { text: "—", sort: -1, title: "No GPU VRAM metrics" };
    }
    const gib = mib / 1024;
    const mean = gpu?.mean_peak_used_mib;
    let title =
      mean != null
        ? `Peak ${mib} MiB (${gib.toFixed(2)} GiB) · mean case peak ${mean.toFixed(0)} MiB`
        : `Peak ${mib} MiB (${gib.toFixed(2)} GiB)`;
    const est =
      gpu?.source === "vram_probe_2case" || model.metrics_backfill?.vram === "vram_probe_2case";
    if (est) {
      title += " · estimated from 2-case VRAM probe (ReAct + direct)";
    }
    return {
      text: est ? `${gib.toFixed(1)}†` : `${gib.toFixed(1)}`,
      sort: mib,
      title,
      estimated: est,
    };
  }

  /**
   * @param {ModelRow} model
   * @param {string} key
   */
  function sortValue(model, key) {
    if (key === "name") {
      return model.name.toLowerCase();
    }
    if (key === "params") {
      return (model.params || "").toLowerCase();
    }
    if (key === "total_pct") {
      return pct(model.total_passed, model.total_cases);
    }
    if (key === "runtime_wall_s") {
      return runtimeCell(model).sort;
    }
    if (key === "vram_peak_mib") {
      return vramCell(model).sort;
    }
    if (key === "completion_tok_per_s") {
      return completionTokCell(model).sort;
    }
    const cell = dimCell(model, key);
    return cell.sort;
  }

  function renderHeader() {
    const thead = table.querySelector("thead tr");
    if (!thead) {
      return;
    }
    thead.innerHTML = "";

    /** @param {string} key, @param {string} label, @param {boolean} numeric */
    function addHeader(key, label, numeric = true) {
      const th = document.createElement("th");
      th.scope = "col";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "eval-sort-btn";
      btn.dataset.sortKey = key;
      btn.textContent = label;
      if (sortState.key === key) {
        btn.dataset.sortDir = sortState.dir;
        btn.setAttribute(
          "aria-sort",
          sortState.dir === "asc" ? "ascending" : "descending",
        );
      } else {
        btn.setAttribute("aria-sort", "none");
      }
      th.appendChild(btn);
      if (numeric) {
        th.classList.add("eval-num");
      }
      thead.appendChild(th);
    }

    addHeader("name", "Model", false);
    addHeader("params", "Params", false);
    addHeader("runtime_wall_s", "Runtime");
    addHeader("completion_tok_per_s", "Out tok/s");
    addHeader("vram_peak_mib", "Max VRAM (GiB)");
    addHeader("total_pct", "Total %");
    for (const dim of dimensions) {
      addHeader(dim.id, dim.label);
    }
  }

  function renderBody() {
    tbody.innerHTML = "";
    const sorted = [...models].sort((a, b) => {
      const av = sortValue(a, sortState.key);
      const bv = sortValue(b, sortState.key);
      let cmp = 0;
      if (typeof av === "string" && typeof bv === "string") {
        cmp = av.localeCompare(bv);
      } else {
        cmp = Number(av) - Number(bv);
      }
      return sortState.dir === "asc" ? cmp : -cmp;
    });

    for (const model of sorted) {
      const tr = document.createElement("tr");
      tr.tabIndex = 0;
      tr.className = "eval-model-row";
      tr.dataset.modelId = model.id;
      tr.setAttribute("role", "button");
      tr.setAttribute(
        "aria-label",
        `View transcripts for ${model.name}. ${model.total_passed} of ${model.total_cases} passed.`,
      );

      function td(text, className, title) {
        const cell = document.createElement("td");
        cell.textContent = text;
        if (className) {
          cell.className = className;
        }
        if (title) {
          cell.title = title;
        }
        return cell;
      }

      tr.appendChild(td(model.name, "eval-model-name"));
      tr.appendChild(td(model.params || "—", ""));

      const rt = runtimeCell(model);
      const rtTd = td(rt.text, "eval-num eval-metric", rt.title);
      if (rt.estimated) {
        rtTd.classList.add("eval-metric-estimated");
      }
      tr.appendChild(rtTd);
      const tok = completionTokCell(model);
      tr.appendChild(td(tok.text, "eval-num eval-metric", tok.title));
      const vr = vramCell(model);
      const vrTd = td(vr.text, "eval-num eval-metric", vr.title);
      if (vr.estimated) {
        vrTd.classList.add("eval-metric-estimated");
      }
      tr.appendChild(vrTd);

      /** @param {string} dimId */
      function appendDimTd(dimId) {
        const cell = dimCell(model, dimId);
        const el = td(cell.text, "eval-num", cell.title);
        if (cell.sort >= 0 && cell.sort < 50) {
          el.classList.add("eval-score-low");
        } else if (cell.sort >= 80) {
          el.classList.add("eval-score-high");
        }
        tr.appendChild(el);
      }

      const totalPct = pct(model.total_passed, model.total_cases);
      tr.appendChild(
        td(`${totalPct}%`, "eval-num", `${model.total_passed}/${model.total_cases}`),
      );
      for (const dim of dimensions) {
        appendDimTd(dim.id);
      }

      tbody.appendChild(tr);
    }
  }

  /**
   * Lightweight markdown: fenced code blocks + inline code + paragraphs.
   * @param {string} raw
   */
  function formatAnswer(raw) {
    const container = document.createElement("div");
    container.className = "eval-formatted-answer";
    if (!raw) {
      container.innerHTML = '<p class="eval-empty">(empty response)</p>';
      return container;
    }

    const fence = /```(\w*)\n([\s\S]*?)```/g;
    let last = 0;
    let match = fence.exec(raw);
    while (match) {
      if (match.index > last) {
        appendTextFragment(container, raw.slice(last, match.index));
      }
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (match[1]) {
        code.className = `language-${match[1]}`;
      }
      code.textContent = match[2].replace(/\n$/, "");
      pre.appendChild(code);
      container.appendChild(pre);
      last = match.index + match[0].length;
      match = fence.exec(raw);
    }
    if (last < raw.length) {
      appendTextFragment(container, raw.slice(last));
    }
    return container;
  }

  /**
   * @param {HTMLElement} parent
   * @param {string} text
   */
  function appendTextFragment(parent, text) {
    const parts = text.split(/(`[^`]+`)/g);
    const p = document.createElement("p");
    for (const part of parts) {
      if (part.startsWith("`") && part.endsWith("`")) {
        const code = document.createElement("code");
        code.textContent = part.slice(1, -1);
        p.appendChild(code);
      } else if (part) {
        p.appendChild(document.createTextNode(part));
      }
    }
    if (p.childNodes.length) {
      parent.appendChild(p);
    }
  }

  /**
   * @param {ModelRow} model
   */
  function casesByDimension(model) {
    const byDim = new Map();
    for (const c of model.cases || []) {
      if (!byDim.has(c.dimension)) {
        byDim.set(c.dimension, []);
      }
      byDim.get(c.dimension).push(c);
    }
    return byDim;
  }

  /**
   * @param {ModelRow} model
   * @param {{ id: string, label: string }} dim
   */
  function dialogDimButtonLabel(model, dim) {
    const block = model.by_dimension[dim.id];
    if (!block?.total) {
      return dim.label;
    }
    return `${dim.label} (${block.passed}/${block.total})`;
  }

  function renderDialogViewBar() {
    if (!dialogViewBar || !dialogModel) {
      return;
    }
    dialogViewBar.hidden = false;
    dialogViewBar.innerHTML = "";
    const label = document.createElement("span");
    label.className = "eval-view-bar-label";
    label.textContent = "Show";
    dialogViewBar.appendChild(label);

    for (const dim of dimensions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "eval-view-btn";
      btn.dataset.viewId = dim.id;
      btn.textContent = dialogDimButtonLabel(dialogModel, dim);
      if (dialogActiveView === dim.id) {
        btn.setAttribute("aria-current", "true");
      }
      dialogViewBar.appendChild(btn);
    }
  }

  function renderDialogBody() {
    if (!dialogModel || !dialogBody) {
      return;
    }
    dialogBody.innerHTML = "";
    const dim = dimensions.find((d) => d.id === dialogActiveView) || dimensions[0];
    const sub = document.createElement("p");
    sub.className = "eval-dialog-sub";
    const block = dialogModel.by_dimension[dim?.id || ""];
    const dimScore = block?.total ? `${block.passed}/${block.total} in ${dim?.label}` : dim?.label;
    sub.textContent = `${dialogModel.ollama_tag || ""} · ${dimScore} · Esc to close`;
    dialogBody.appendChild(sub);

    const byDim = casesByDimension(dialogModel);
    const cases = byDim.get(dim?.id || "") || [];
    if (!cases.length) {
      const empty = document.createElement("p");
      empty.className = "eval-empty";
      empty.textContent = "No cases for this dimension.";
      dialogBody.appendChild(empty);
      return;
    }

    const section = document.createElement("section");
    section.className = "eval-case-section";
    for (const c of cases) {
      const article = document.createElement("article");
      article.className = c.passed ? "eval-case eval-case-pass" : "eval-case eval-case-fail";

      const head = document.createElement("div");
      head.className = "eval-case-head";
      head.innerHTML = `<strong>${c.case_id}</strong> <span class="eval-case-status">${c.passed ? "PASS" : "FAIL"}</span>`;
        if (c.failure_modes?.length) {
          head.innerHTML += ` <span class="eval-fail-modes">${c.failure_modes.join(", ")}</span>`;
        }
        const infra = c.signals?.infrastructure;
        if (infra?.categories?.length) {
          head.innerHTML += ` <span class="eval-fail-modes eval-infra-tags">harness: ${infra.categories.join(", ")}</span>`;
        }
      article.appendChild(head);

      const promptLabel = document.createElement("div");
      promptLabel.className = "eval-label";
      promptLabel.textContent = "Prompt";
      article.appendChild(promptLabel);
      const promptPre = document.createElement("pre");
      promptPre.className = "eval-prompt";
      promptPre.textContent = c.prompt;
      article.appendChild(promptPre);

      const ansLabel = document.createElement("div");
      ansLabel.className = "eval-label";
      ansLabel.textContent = "Model output";
      article.appendChild(ansLabel);
      article.appendChild(formatAnswer(c.final_answer || ""));

      section.appendChild(article);
    }
    dialogBody.appendChild(section);
    dialogBody.scrollTop = 0;
  }

  /**
   * @param {ModelRow} model
   * @param {string} [preferredView]
   */
  function openModelDialog(model, preferredView) {
    dialogModel = model;
    dialogTitle.textContent = model.name;
    const byDim = casesByDimension(model);
    if (preferredView && byDim.get(preferredView)?.length) {
      dialogActiveView = preferredView;
    } else if (!byDim.get(dialogActiveView)?.length) {
      const firstWithCases = dimensions.find((d) => byDim.get(d.id)?.length);
      dialogActiveView = firstWithCases?.id || dimensions[0]?.id || "tool_use";
    }
    renderDialogViewBar();
    renderDialogBody();

    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    }
  }

  /**
   * @param {string} modelId
   */
  function findModel(modelId) {
    return models.find((m) => m.id === modelId);
  }

  dialogViewBar?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const btn = target.closest(".eval-view-btn");
    if (!(btn instanceof HTMLButtonElement)) {
      return;
    }
    const viewId = btn.dataset.viewId;
    if (!viewId || viewId === dialogActiveView) {
      return;
    }
    dialogActiveView = viewId;
    renderDialogViewBar();
    renderDialogBody();
  });

  document.addEventListener("agents-model-eval-open", (event) => {
    const detail = /** @type {CustomEvent} */ (event).detail;
    const model = findModel(detail?.modelId || "");
    if (model) {
      openModelDialog(model);
    }
  });

  table.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const sortBtn = target.closest(".eval-sort-btn");
    if (sortBtn instanceof HTMLButtonElement) {
      const key = sortBtn.dataset.sortKey || "total_pct";
      if (sortState.key === key) {
        sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
      } else {
        sortState = { key, dir: "desc" };
      }
      renderHeader();
      renderBody();
      return;
    }

    const row = target.closest(".eval-model-row");
    if (row instanceof HTMLTableRowElement) {
      const model = findModel(row.dataset.modelId || "");
      if (model) {
        openModelDialog(model);
      }
    }
  });

  tbody.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const row = event.target;
    if (!(row instanceof HTMLTableRowElement) || !row.classList.contains("eval-model-row")) {
      return;
    }
    event.preventDefault();
    const model = findModel(row.dataset.modelId || "");
    if (model) {
      openModelDialog(model);
    }
  });

  const closeBtn = dialog.querySelector("[data-action='close-eval-dialog']");
  closeBtn?.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    dialogModel = null;
    if (dialogViewBar) {
      dialogViewBar.hidden = true;
    }
  });

  renderHeader();
  renderBody();
})();
