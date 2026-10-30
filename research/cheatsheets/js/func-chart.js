/**
 * Mini SVG line charts for cheatsheet cells.
 * Usage: <div class="func-chart" data-func-chart="gelu,silu" data-chart-labels="GELU,SiLU"></div>
 */
(function () {
  const DEFAULT_X = [-4, 4];
  const SAMPLES = 160;

  /** @param {number} x */
  function sigmoid(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  /** @param {number} x */
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
        t *
        Math.exp(-ax * ax);
    return sign * y;
  }

  /** @type {Record<string, (x: number) => number>} */
  const FUNCS = {
    relu: (x) => Math.max(0, x),
    gelu: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
    quickgelu: (x) => x * sigmoid(1.702 * x),
    silu: (x) => x * sigmoid(x),
    mish: (x) => x * Math.tanh(Math.log1p(Math.exp(x))),
    relu2: (x) => {
      const t = Math.max(0, x);
      return t * t;
    },
    sigmoid: (x) => sigmoid(x),
    powlu: (x) => {
      const m = 3;
      const s = sigmoid(x);
      if (x > 0) {
        return x * Math.pow(x, m / (Math.sqrt(x) + 1)) * s;
      }
      return x * x * s;
    },
    silu_clamp: (x) => {
      const c = 3;
      const t = Math.max(-c, Math.min(c, x));
      return t * sigmoid(t);
    },
  };

  /**
   * @param {string} name
   * @returns {(x: number) => number}
   */
  function resolveFunc(name) {
    const key = name.trim().toLowerCase();
    const fn = FUNCS[key];
    if (!fn) {
      return () => 0;
    }
    return fn;
  }

  /**
   * @param {number[]} xs
   * @param {((x: number) => number)[]} fns
   */
  function sampleSeries(xs, fns) {
    return fns.map((fn) => xs.map((x) => fn(x)));
  }

  /**
   * @param {HTMLElement} el
   */
  function renderChart(el) {
    const names = (el.dataset.funcChart || "relu")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const labels = (el.dataset.chartLabels || names.join(",")).split(",").map((s) => s.trim());
    const xMin = Number(el.dataset.xMin ?? DEFAULT_X[0]);
    const xMax = Number(el.dataset.xMax ?? DEFAULT_X[1]);
    const yMinOverride = el.dataset.yMin !== undefined ? Number(el.dataset.yMin) : null;
    const yMaxOverride = el.dataset.yMax !== undefined ? Number(el.dataset.yMax) : null;

    const fns = names.map(resolveFunc);
    const xs = Array.from({ length: SAMPLES }, (_, i) => xMin + ((xMax - xMin) * i) / (SAMPLES - 1));
    const series = sampleSeries(xs, fns);

    let yMin = Infinity;
    let yMax = -Infinity;
    for (const ys of series) {
      for (const y of ys) {
        if (Number.isFinite(y)) {
          yMin = Math.min(yMin, y);
          yMax = Math.max(yMax, y);
        }
      }
    }
    if (yMinOverride !== null) {
      yMin = yMinOverride;
    }
    if (yMaxOverride !== null) {
      yMax = yMaxOverride;
    }
    const yPad = (yMax - yMin) * 0.08 || 0.5;
    yMin -= yPad;
    yMax += yPad;

    const W = 280;
    const H = 100;
    const pad = { l: 26, r: 8, t: 10, b: 16 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    /** @param {number} x */
    const mapX = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * plotW;
    /** @param {number} y */
    const mapY = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    const zeroY = mapY(0);
    const zeroX = mapX(0);

    const strokeColors = [
      "var(--chart-line-1)",
      "var(--chart-line-2)",
      "var(--chart-line-3)",
    ];

    const paths = series
      .map((ys, idx) => {
        const parts = ys.map((y, i) => {
          const cmd = i === 0 ? "M" : "L";
          return `${cmd}${mapX(xs[i]).toFixed(2)},${mapY(y).toFixed(2)}`;
        });
        return `<path d="${parts.join(" ")}" fill="none" stroke="${strokeColors[idx % strokeColors.length]}" stroke-width="2" vector-effect="non-scaling-stroke" />`;
      })
      .join("");

    const legend =
      names.length > 1
        ? `<g class="func-chart-legend">${names
            .map((name, idx) => {
              const lx = pad.l + idx * 72;
              return `<line x1="${lx}" y1="${H - 4}" x2="${lx + 14}" y2="${H - 4}" stroke="${strokeColors[idx % strokeColors.length]}" stroke-width="2"/><text x="${lx + 18}" y="${H - 1}" font-size="8" fill="var(--text-muted)">${labels[idx] || name}</text>`;
            })
            .join("")}</g>`
        : "";

    const ariaLabel =
      el.getAttribute("aria-label") ||
      `Plot of ${labels.join(", ")} from ${xMin} to ${xMax}`;

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ariaLabel.replace(/"/g, "&quot;")}" xmlns="http://www.w3.org/2000/svg">
      <rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" fill="var(--chart-bg)" />
      <line x1="${pad.l}" y1="${zeroY}" x2="${pad.l + plotW}" y2="${zeroY}" stroke="var(--chart-axis)" stroke-width="1" />
      <line x1="${zeroX}" y1="${pad.t}" x2="${zeroX}" y2="${pad.t + plotH}" stroke="var(--chart-axis)" stroke-width="1" />
      <text x="${pad.l + plotW - 2}" y="${zeroY - 3}" text-anchor="end" font-size="7" fill="var(--text-muted)">0</text>
      ${paths}
      ${legend}
    </svg>`;
  }

  function init() {
    document.querySelectorAll(".func-chart[data-func-chart]").forEach(renderChart);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
