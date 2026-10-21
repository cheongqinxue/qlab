import { test, expect } from "@playwright/test";
import {
  addPanel,
  addSection,
  clearWorkspaceStorage,
  focusFirstRun,
  waitForRunsLoaded,
} from "../helpers.js";

test.describe("Metric matrix (tile heatmap)", () => {
  test.beforeEach(async ({ page }) => {
    await clearWorkspaceStorage(page);
    await page.goto("/");
    await waitForRunsLoaded(page);
    await focusFirstRun(page);
    const hasAttention = await page
      .getByTestId("metric-family-train--attention_max_weight")
      .isVisible()
      .catch(() => false);
    test.skip(
      !hasAttention,
      "Llama update_rank fixture has no attention_max_weight; set BEV_E2E_SCAN_ROOT=e2e_minimal",
    );
    await addSection(page);
  });

  test("renders tiles after batch series load", async ({ page }) => {
    await addPanel(page, "Metric matrix");
    await page.getByRole("button", { name: "Settings" }).click();
    const batchPromise = page.waitForResponse(
      (response) => response.url().includes("/series/batch") && response.ok(),
      { timeout: 30_000 },
    );
    await page.getByRole("button", { name: "Apply selection" }).click();
    await batchPromise;
    await expect(page.getByText(/Loading series/)).toHaveCount(0, {
      timeout: 30_000,
    });
    const tile = page.getByTestId("multipanel-tile").first();
    await expect(tile).toBeVisible({ timeout: 15_000 });
    const box = await tile.boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual(8);
    expect(box?.width).toBeLessThanOrEqual(12);
    expect(box?.height).toBeGreaterThanOrEqual(8);
    expect(box?.height).toBeLessThanOrEqual(12);
  });
});
