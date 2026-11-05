/**
 * Shared cheatsheet UI: theme (light/dark), font scale, persist preferences.
 */
(function () {
  const STORAGE_KEY = "qlab-cheatsheet-prefs";

  /** @typedef {{ theme: 'light' | 'dark', fontScale: number }} Prefs */

  /** @returns {Prefs} */
  function loadPrefs() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return { theme: "light", fontScale: 1 };
      }
      const parsed = JSON.parse(raw);
      return {
        theme: parsed.theme === "dark" ? "dark" : "light",
        fontScale: clampScale(parsed.fontScale),
      };
    } catch {
      return { theme: "light", fontScale: 1 };
    }
  }

  /** @param {unknown} value */
  function clampScale(value) {
    const n = typeof value === "number" ? value : 1;
    if (n <= 0.85) {
      return 0.85;
    }
    if (n >= 1.25) {
      return 1.25;
    }
    return Math.round(n * 100) / 100;
  }

  /** @param {Prefs} prefs */
  function savePrefs(prefs) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  }

  /** @param {Prefs} prefs */
  function applyPrefs(prefs) {
    const root = document.documentElement;
    root.dataset.theme = prefs.theme;
    root.style.setProperty("--font-scale", String(prefs.fontScale));

    const themeBtn = document.querySelector("[data-action='toggle-theme']");
    if (themeBtn) {
      themeBtn.setAttribute("aria-pressed", prefs.theme === "dark" ? "true" : "false");
      themeBtn.textContent = prefs.theme === "dark" ? "Light" : "Dark";
    }

    document.querySelectorAll("[data-action='font-smaller']").forEach((btn) => {
      btn.disabled = prefs.fontScale <= 0.85;
    });
    document.querySelectorAll("[data-action='font-larger']").forEach((btn) => {
      btn.disabled = prefs.fontScale >= 1.25;
    });
  }

  /** Narrow read column when every grid cell is full-width (no two-column .cell pairs). */
  function applySheetLayout() {
    const sheet = document.querySelector(".sheet");
    const grid = document.querySelector(".sheet-grid");
    if (!sheet || !grid) {
      return;
    }
    const hasHalfWidthCell = grid.querySelector(".cell:not(.cell-span):not(.cell-band)");
    if (!hasHalfWidthCell) {
      sheet.dataset.layout = "single";
    }
  }

  function init() {
    applySheetLayout();
    let prefs = loadPrefs();
    applyPrefs(prefs);

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      const action = target.closest("[data-action]")?.getAttribute("data-action");
      if (!action) {
        return;
      }

      if (action === "toggle-theme") {
        prefs = {
          ...prefs,
          theme: prefs.theme === "dark" ? "light" : "dark",
        };
        applyPrefs(prefs);
        savePrefs(prefs);
        return;
      }

      if (action === "font-smaller") {
        prefs = { ...prefs, fontScale: clampScale(prefs.fontScale - 0.1) };
        applyPrefs(prefs);
        savePrefs(prefs);
        return;
      }

      if (action === "font-larger") {
        prefs = { ...prefs, fontScale: clampScale(prefs.fontScale + 0.1) };
        applyPrefs(prefs);
        savePrefs(prefs);
      }
    });

    if (!document.querySelector("[data-cheatsheet-nav]")) {
      const path = window.location.pathname.split("/").pop() || "index.html";
      document.querySelectorAll(".nav-list a").forEach((link) => {
        const href = link.getAttribute("href");
        if (href === path || (path === "" && href === "index.html")) {
          link.classList.add("is-active");
          link.setAttribute("aria-current", "page");
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
