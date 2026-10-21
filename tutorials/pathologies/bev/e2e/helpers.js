/** Shared Playwright helpers for BEV workspace flows. */

/** Offline Llama-scale fixture run (see ``wandb_llama70b_scale.py``). */
export const LLAMA70B_RUN_NAME = "llama70b-update-rank-scale";

/** Single plottable key on the Llama fixture (no head suffix). */
export const LLAMA_SAMPLE_METRIC_KEY = "train/update_rank/model.embed_tokens.weight";

/** Regex for a self-attn multipanel slice (layer 0 q/k/v/o weights). */
export const LLAMA_SELF_ATTN_PATH_REGEX =
  String.raw`model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight`;

/**
 * @param {import('@playwright/test').Page} page
 */
export async function clearWorkspaceStorage(page) {
  await page.addInitScript(() => {
    localStorage.clear();
  });
}

/**
 * Reset server-side workspace document (survives reload; not in localStorage).
 *
 * @param {import('@playwright/test').Page} page
 */
export async function resetServerWorkspace(page) {
  await page.evaluate(async () => {
    await fetch("/api/workspace", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sections: [] }),
    });
  });
}

/**
 * @param {import('@playwright/test').Page} page
 */
export async function waitForRunsLoaded(page) {
  await page.getByTestId("run-sidebar").waitFor({ state: "visible" });
  await page.locator(".run-row-main").first().waitFor({ state: "visible" });
}

/**
 * @param {import('@playwright/test').Page} page
 */
export async function focusFirstRun(page) {
  await page.locator(".run-row-main").first().click();
  await page.locator(".run-row.focused").waitFor({ state: "visible" });
}

/**
 * @param {import('@playwright/test').Page} page
 */
export async function addSection(page) {
  await page.getByTestId("workspace-add-section").click();
  await page.locator(".workspace-section").first().waitFor({ state: "visible" });
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} label Menu label (e.g. "Attention heatmap").
 */
export async function addPanel(page, label) {
  await page.getByRole("button", { name: "Add panel" }).first().click();
  await page.getByRole("menuitem", { name: label }).click();
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} title Panel title shown in the header input.
 */
export async function openPanelSettings(page, title) {
  await page.getByRole("button", { name: `${title} settings` }).click();
}

/**
 * Expand metric key tree branches so a leaf with the given full key is visible.
 *
 * @param {import('@playwright/test').Page} page
 * @param {string} metricKey
 * @param {import('@playwright/test').Locator} [scope]
 */
export async function revealMetricKey(page, metricKey, scope = page.locator("body")) {
  const parts = metricKey.split("/");
  let path = "";
  for (let index = 0; index < parts.length - 1; index += 1) {
    path = path ? `${path}/${parts[index]}` : parts[index];
    const branch = scope.locator(`button.metric-key-tree-branch:has([title="${path}"])`).first();
    const expanded = await branch.getAttribute("aria-expanded");
    if (expanded === "false") {
      await branch.click();
    }
  }
}

/**
 * @param {import('@playwright/test').Page} page
 * @param {string} title
 */
export function panelByTitle(page, title) {
  return page.locator("section.panel", {
    has: page.getByRole("button", { name: `${title} settings` }),
  });
}

/**
 * @param {import('@playwright/test').Locator} panel
 * @param {string} metricKey
 * @param {"checkbox" | "radio"} inputType
 */
export async function pickMetricInPanel(page, panel, metricKey, inputType = "checkbox") {
  const leafLabel = metricKey.split("/").pop() ?? metricKey;
  await revealMetricKey(page, metricKey, panel);
  await panel.getByRole(inputType, { name: leafLabel }).check();
}
