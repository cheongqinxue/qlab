/**
 * Harness failure mode counts (from saved JSONL / rescore_infrastructure.py).
 */
(function () {
  const data = window.AGENTS_MODEL_EVAL_DATA;
  const table = document.getElementById("eval-infra-matrix");
  const tbody = document.getElementById("eval-infra-matrix-body");
  if (!data || !table || !tbody) {
    return;
  }

  /** @type {{ id: string, label: string }[]} */
  const infraModes = data.infrastructure_failure_modes || [];
  /** @type {Array<{ id: string, name: string, infrastructure_failure_counts?: Record<string, number> }>} */
  const models = data.models || [];
  /** @type {{ key: string, dir: 'asc' | 'desc' }} */
  let sortState = { key: "infra_total", dir: "desc" };

  /**
   * @param {Record<string, number>} counts
   * @param {string} modeId
   */
  function countFor(counts, modeId) {
    const n = counts?.[modeId];
    return typeof n === "number" ? n : 0;
  }

  /**
   * @param {Record<string, number>} counts
   */
  function infraTotal(counts) {
    let sum = 0;
    for (const mode of infraModes) {
      sum += countFor(counts, mode.id);
    }
    return sum;
  }

  function renderHeader() {
    const thead = table.querySelector("thead tr");
    if (!thead) {
      return;
    }
    thead.innerHTML = "";

    /** @param {string} key @param {string} label @param {boolean} numeric */
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
    addHeader("infra_total", "Tags Σ");
    for (const mode of infraModes) {
      addHeader(mode.id, mode.label);
    }
  }

  /**
   * @param {typeof models[0]} model
   * @param {string} key
   */
  function sortValue(model, key) {
    const counts = model.infrastructure_failure_counts || {};
    if (key === "name") {
      return model.name.toLowerCase();
    }
    if (key === "infra_total") {
      return infraTotal(counts);
    }
    return countFor(counts, key);
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
      const counts = model.infrastructure_failure_counts || {};
      const tr = document.createElement("tr");
      const total = infraTotal(counts);

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
      const totalCell = td(String(total), "eval-num eval-infra-total");
      if (total > 0) {
        totalCell.classList.add("eval-infra-has-failures");
      }
      tr.appendChild(totalCell);

      for (const mode of infraModes) {
        const n = countFor(counts, mode.id);
        const cell = td(n ? String(n) : "—", "eval-num");
        if (n > 0) {
          cell.classList.add("eval-infra-hit");
          cell.title = `${mode.label}: ${n} failed case(s)`;
        }
        tr.appendChild(cell);
      }
      tbody.appendChild(tr);
    }
  }

  table.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const sortBtn = target.closest(".eval-sort-btn");
    if (!(sortBtn instanceof HTMLButtonElement)) {
      return;
    }
    const key = sortBtn.dataset.sortKey || "infra_total";
    if (sortState.key === key) {
      sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
    } else {
      sortState = { key, dir: "desc" };
    }
    renderHeader();
    renderBody();
  });

  renderHeader();
  renderBody();
})();
