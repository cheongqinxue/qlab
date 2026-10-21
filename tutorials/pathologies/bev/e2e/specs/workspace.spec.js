import { test, expect } from "@playwright/test";
import {
  addPanel,
  addSection,
  clearWorkspaceStorage,
  focusFirstRun,
  LLAMA_SAMPLE_METRIC_KEY,
  LLAMA_SELF_ATTN_PATH_REGEX,
  openPanelSettings,
  panelByTitle,
  pickMetricInPanel,
  waitForRunsLoaded,
  resetServerWorkspace,
} from "../helpers.js";

test.describe("Workspace panels", () => {
  test.beforeEach(async ({ page }) => {
    await clearWorkspaceStorage(page);
    await page.goto("/");
    await resetServerWorkspace(page);
    await page.reload();
    await waitForRunsLoaded(page);
    await focusFirstRun(page);
    await addSection(page);
  });

  test("adds a line panel and plots a selected metric", async ({ page }) => {
    await addPanel(page, "Line plot");
    await openPanelSettings(page, "Line plot");
    const linePanel = panelByTitle(page, "Line plot");
    await linePanel
      .locator('label:has-text("Key regex") input')
      .fill(String.raw`^train/update_rank/model\.embed_tokens\.weight$`);
    await page.waitForTimeout(800);
    await expect(linePanel.locator(".uplot")).toBeVisible({ timeout: 120_000 });
  });

  test("adds a multipanel from metric scope", async ({ page }) => {
    await addPanel(page, "Multipanel");
    await openPanelSettings(page, "Multipanel");
    const multipanel = panelByTitle(page, "Multipanel");
    await multipanel.getByRole("combobox", { name: "Metric" }).selectOption("train/update_rank");
    await multipanel.getByRole("textbox", { name: "Path regex" }).fill("model\\.layers\\.\\d+\\.mlp");
    await page.waitForTimeout(1500);
    const batchDone = page.waitForResponse(
      (response) => response.url().includes("/series/batch") && response.ok(),
      { timeout: 120_000 },
    );
    await multipanel.getByRole("button", { name: "Apply selection" }).click();
    await batchDone;
    await expect(multipanel.locator(".multipanel-host .uplot").first()).toBeVisible({
      timeout: 120_000,
    });
  });

  test("adds a scalar readout for one metric", async ({ page }) => {
    await page.evaluate(
      async (metricKey) => {
        const section = {
          id: "e2e-scalar-section",
          name: "Scalar section",
          isOpen: true,
          columns: 1,
          panels: [
            {
              id: "e2e-scalar-panel",
              viz: "scalar",
              title: "Scalar",
              metricKeys: [metricKey],
            },
          ],
        };
        await fetch("/api/workspace", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sections: [section] }),
        });
      },
      LLAMA_SAMPLE_METRIC_KEY,
    );
    await page.reload();
    await waitForRunsLoaded(page);
    await focusFirstRun(page);
    const scalarPanel = panelByTitle(page, "Scalar");
    await expect(scalarPanel.locator(".scalar-readout-compact")).toBeVisible({
      timeout: 120_000,
    });
    await expect(scalarPanel.locator(".scalar-value")).not.toHaveText("");
  });
});

test.describe("Workspace multipanel (Llama self_attn)", () => {
  test.beforeEach(async ({ page }) => {
    await clearWorkspaceStorage(page);
    await page.goto("/");
    await resetServerWorkspace(page);
    await page.reload();
    await waitForRunsLoaded(page);
    await focusFirstRun(page);
    await addSection(page);
    await page.setViewportSize({ width: 1280, height: 900 });
  });

  test("every mini-chart cell has seamless grid borders", async ({ page }) => {
    await addPanel(page, "Multipanel");
    await openPanelSettings(page, "Multipanel");
    const multipanel = panelByTitle(page, "Multipanel");
    await multipanel.getByRole("combobox", { name: "Metric" }).selectOption("train/update_rank");
    await multipanel.getByRole("textbox", { name: "Path regex" }).fill(LLAMA_SELF_ATTN_PATH_REGEX);
    await page.waitForTimeout(1500);
    const batchDone = page.waitForResponse(
      (response) => response.url().includes("/series/batch") && response.ok(),
      { timeout: 120_000 },
    );
    await multipanel.getByRole("button", { name: "Apply selection" }).click();
    await batchDone;
    await expect(multipanel.locator(".multipanel-host .uplot").first()).toBeVisible({
      timeout: 120_000,
    });
    await page.waitForTimeout(400);

    const cells = multipanel.locator(".multipanel-row-charts .multiple-seamless");
    const count = await cells.count();
    expect(count).toBeGreaterThan(8);

    for (let index = 0; index < count; index += 1) {
      const plot = cells.nth(index).locator(".chart-wrap-seamless .uplot");
      await expect(plot, `cell ${index} uPlot`).toBeVisible();
    }
  });
});
