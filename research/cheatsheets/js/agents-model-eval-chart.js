/**
 * Pass rate vs runtime/throughput scatter with Pareto frontier (model eval cheatsheet).
 */
(function () {
  const data = window.AGENTS_MODEL_EVAL_DATA;
  const root = document.getElementById("eval-pareto-chart");
  const noteEl = document.getElementById("eval-pareto-note");
  const xToggle = document.getElementById("eval-pareto-x-mode");
  if (!data || !root) {
    return;
  }

  /** @type {'runtime' | 'throughput' | 'completion_tok_s'} */
  let xMode = "runtime";
  /**
   * @param {number} passed
   * @param {number} total
   */
  function pct(passed, total) {
    if (!total) {
      return null;
    }
    return Math.round((1000 * passed) / total) / 10;
  }

  /**
   * @param {object} model
   */
  function passRateY(model) {
    return pct(model.total_passed, model.total_cases);
  }

  /**
   * @param {object} model
   */
  function xValue(model) {
    if (xMode === "completion_tok_s") {
      const tps = model.token_stats?.completion_tok_per_s;
      if (tps == null || !Number.isFinite(tps) || tps <= 0) {
        return null;
      }
      return tps;
    }
    const wall = model.runtime?.wall_s;
    if (wall == null || !Number.isFinite(wall) || wall <= 0) {
      return null;
    }
    if (xMode === "throughput") {
      const cases = model.total_cases || 0;
      return (cases / wall) * 60;
    }
    return wall / 60;
  }

  /**
   * @typedef {{ model: object, x: number, y: number, id: string, label: string }} PlotPoint
   */

  /** @returns {PlotPoint[]} */
  function buildPoints() {
    /** @type {PlotPoint[]} */
    const points = [];
    for (const model of data.models || []) {
      const x = xValue(model);
      const y = passRateY(model);
      if (x == null || y == null) {
        continue;
      }
      points.push({
        model,
        x,
        y,
        id: model.id,
        label: model.name,
      });
    }
    return points;
  }

  /**
   * Non-dominated points: maximize Y; X lower is better (runtime) or higher (throughput).
   * @param {PlotPoint[]} points
   */
  function paretoFrontier(points) {
    const xHigherIsBetter = xMode === "throughput" || xMode === "completion_tok_s";
    /** @type {PlotPoint[]} */
    const frontier = [];
    for (const p of points) {
      let dominated = false;
      for (const q of points) {
        if (p === q) {
          continue;
        }
        if (xHigherIsBetter) {
          if (q.y >= p.y && q.x >= p.x && (q.y > p.y || q.x > p.x)) {
            dominated = true;
            break;
          }
        } else if (q.y >= p.y && q.x <= p.x && (q.y > p.y || q.x < p.x)) {
          dominated = true;
          break;
        }
      }
      if (!dominated) {
        frontier.push(p);
      }
    }
    return frontier.sort((a, b) => a.x - b.x);
  }

  /**
   * @param {number} value
   * @param {number} digits
   */
  function fmtNum(value, digits = 1) {
    if (!Number.isFinite(value)) {
      return "";
    }
    if (Math.abs(value) >= 100) {
      return String(Math.round(value));
    }
    return value.toFixed(digits);
  }

  /**
   * @param {number} n
   * @param {number} lo
   * @param {number} hi
   */
  function clamp(n, lo, hi) {
    return Math.min(hi, Math.max(lo, n));
  }

  function render() {
    const points = buildPoints();
    const frontier = paretoFrontier(points);
    const rect = root.getBoundingClientRect();
    const W = Math.max(320, Math.round(rect.width) || 640);
    const H = Math.max(260, Math.min(420, Math.round(W * 0.52)));

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of points) {
      xMin = Math.min(xMin, p.x);
      xMax = Math.max(xMax, p.x);
      yMin = Math.min(yMin, p.y);
      yMax = Math.max(yMax, p.y);
    }
    if (!points.length) {
      root.innerHTML =
        '<p class="eval-pareto-empty">No models with both runtime and scores for this view.</p>';
      if (noteEl) {
        noteEl.textContent = "";
      }
      return;
    }

    yMin = clamp(yMin - 5, 0, 100);
    yMax = clamp(yMax + 5, 0, 100);
    const xPad = (xMax - xMin) * 0.1 || 1;
    xMin = Math.max(0, xMin - xPad);
    xMax += xPad;
    if (yMax - yMin < 8) {
      yMin = clamp(yMin - 4, 0, 100);
      yMax = clamp(yMax + 4, 0, 100);
    }

    const pad = { l: 52, r: 24, t: 20, b: 44 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    /** @param {number} x */
    const mapX = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * plotW;
    /** @param {number} y */
    const mapY = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    const yLabel = "Total pass rate (%)";
    const xLabel =
      xMode === "completion_tok_s"
        ? "Completion throughput (tok/s)"
        : xMode === "throughput"
          ? "Harness throughput (cases / min)"
          : "Suite runtime (minutes)";

    const frontierPath =
      frontier.length >= 2
        ? `<path class="eval-pareto-line" d="${frontier
            .map((p, i) => `${i === 0 ? "M" : "L"}${mapX(p.x).toFixed(1)},${mapY(p.y).toFixed(1)}`)
            .join(" ")}" fill="none" stroke="var(--chart-line-2)" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke" />`
        : "";

    const frontierIds = new Set(frontier.map((p) => p.id));

    const dots = points
      .map((p) => {
        const cx = mapX(p.x);
        const cy = mapY(p.y);
        const onFrontier = frontierIds.has(p.id);
        const wall = p.model.runtime?.wall_s;
        const tip =
          xMode === "completion_tok_s"
            ? `${p.label}\n${yLabel.split(" (")[0]}: ${p.y}%\nOut: ${fmtNum(p.x, 1)} tok/s`
            : xMode === "throughput"
              ? `${p.label}\n${yLabel.split(" (")[0]}: ${p.y}%\nThroughput: ${fmtNum(p.x, 2)} cases/min\nWall: ${fmtNum(wall / 60, 1)} min`
              : `${p.label}\n${yLabel.split(" (")[0]}: ${p.y}%\nRuntime: ${fmtNum(p.x, 1)} min`;
        return `<g class="eval-pareto-point${onFrontier ? " eval-pareto-point-frontier" : ""}" data-model-id="${p.id}" tabindex="0" role="button" aria-label="${p.label.replace(/"/g, "")}">
          <title>${tip.replace(/\n/g, " · ")}</title>
          <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${onFrontier ? 6 : 5}" class="eval-pareto-dot" />
        </g>`;
      })
      .join("");

    const xTicks = 5;
    const yTicks = 5;
    let grid = "";
    for (let i = 0; i <= xTicks; i += 1) {
      const t = xMin + ((xMax - xMin) * i) / xTicks;
      const px = mapX(t);
      grid += `<line x1="${px}" y1="${pad.t}" x2="${px}" y2="${pad.t + plotH}" class="eval-pareto-grid" />`;
      grid += `<text x="${px}" y="${H - 8}" text-anchor="middle" class="eval-pareto-tick">${fmtNum(t, xMode === "throughput" ? 2 : xMode === "completion_tok_s" ? 0 : 1)}</text>`;
    }
    for (let i = 0; i <= yTicks; i += 1) {
      const t = yMin + ((yMax - yMin) * i) / yTicks;
      const py = mapY(t);
      grid += `<line x1="${pad.l}" y1="${py}" x2="${pad.l + plotW}" y2="${py}" class="eval-pareto-grid" />`;
      grid += `<text x="${pad.l - 8}" y="${py + 3}" text-anchor="end" class="eval-pareto-tick">${fmtNum(t, 0)}</text>`;
    }

    root.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="eval-pareto-svg" role="img" aria-label="Scatter of pass rate versus ${xLabel}" xmlns="http://www.w3.org/2000/svg">
      <rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" class="eval-pareto-plot-bg" />
      ${grid}
      <line x1="${pad.l}" y1="${pad.t + plotH}" x2="${pad.l + plotW}" y2="${pad.t + plotH}" class="eval-pareto-axis" />
      <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + plotH}" class="eval-pareto-axis" />
      ${frontierPath}
      ${dots}
      <text x="${pad.l + plotW / 2}" y="${H - 22}" text-anchor="middle" class="eval-pareto-axis-label">${xLabel}</text>
      <text x="16" y="${pad.t + plotH / 2}" text-anchor="middle" transform="rotate(-90 16 ${pad.t + plotH / 2})" class="eval-pareto-axis-label">${yLabel}</text>
    </svg>`;

    if (noteEl) {
      const xWord =
        xMode === "completion_tok_s"
          ? "out tok/s"
          : xMode === "throughput"
            ? "cases/min"
            : "runtime";
      const xDir = xMode === "runtime" ? "lower" : "higher";
      noteEl.textContent = `${points.length} models · dashed line = Pareto frontier (higher pass, ${xDir} ${xWord}) · click a point for transcripts`;
    }

    root.querySelectorAll(".eval-pareto-point").forEach((g) => {
      g.addEventListener("click", () => {
        const id = g.getAttribute("data-model-id");
        if (id) {
          document.dispatchEvent(
            new CustomEvent("agents-model-eval-open", { detail: { modelId: id } }),
          );
        }
      });
      g.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }
        event.preventDefault();
        g.dispatchEvent(new Event("click"));
      });
    });
  }

  xToggle?.querySelectorAll("[data-x-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-x-mode");
      if (mode !== "runtime" && mode !== "throughput" && mode !== "completion_tok_s") {
        return;
      }
      xMode = mode;
      xToggle.querySelectorAll("[data-x-mode]").forEach((b) => {
        b.setAttribute("aria-current", b === btn ? "true" : "false");
      });
      render();
    });
  });

  new MutationObserver(() => render()).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(render, 120);
  });

  render();
})();
