import { test, expect } from "@playwright/test";
import { clearWorkspaceStorage, LLAMA70B_RUN_NAME, waitForRunsLoaded } from "../helpers.js";

test.describe("BEV smoke", () => {
  test.beforeEach(async ({ page }) => {
    await clearWorkspaceStorage(page);
    await page.goto("/");
    await waitForRunsLoaded(page);
  });

  test("loads runs and focused run manifest in top bar", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "BEV" })).toBeVisible();
    await expect(
      page.getByTestId("run-sidebar").getByRole("button", { name: LLAMA70B_RUN_NAME }),
    ).toBeVisible();
    await page.locator(".run-row-main").first().click();
    await expect(page.locator(".topbar")).toContainText("focus");
    await expect(page.locator(".badge")).toContainText("step");
  });

  test("sidebar lists top-level train metric families", async ({ page }) => {
    await page.locator(".run-row-main").first().click();
    const sidebar = page.locator("aside .metric-tree");
    await expect(sidebar.getByTestId("metric-family-train--update_rank")).toBeVisible();
  });
});
