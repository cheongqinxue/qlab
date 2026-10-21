/**
 * Visual capture for agent/human review (not asserted in CI by default).
 *
 * Run from e2e/:
 *   npm run capture
 *
 * Uses the same Llama-scale scan root as Playwright serve (override with BEV_SCAN_ROOT).
 * Writes PNGs to e2e/screenshots/ (gitignored).
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";
import {
  addPanel,
  addSection,
  clearWorkspaceStorage,
  focusFirstRun,
  LLAMA_SELF_ATTN_PATH_REGEX,
  openPanelSettings,
  panelByTitle,
  waitForRunsLoaded,
} from "../helpers.js";

const specDir = path.dirname(fileURLToPath(import.meta.url));
const screenshotDir = path.join(specDir, "..", "screenshots");

async function configureMultipanel(multipanel, metric, pathRegex) {
  await multipanel.getByRole("combobox", { name: "Metric" }).selectOption(metric);
  if (pathRegex) {
    await multipanel.getByRole("textbox", { name: "Path regex" }).fill(pathRegex);
  }
  await expect(multipanel.locator(".multipanel-host .uplot").first()).toBeVisible({
    timeout: 120_000,
  });
}

test.describe("UI screenshot capture @capture", () => {
  test.beforeAll(() => {
    fs.mkdirSync(screenshotDir, { recursive: true });
  });

  test.beforeEach(async ({ page }) => {
    test.skip(
      !process.env.BEV_CAPTURE_SCREENSHOTS,
      "Set BEV_CAPTURE_SCREENSHOTS=1 (or npm run capture)",
    );
    await clearWorkspaceStorage(page);
    await page.goto("/");
    await waitForRunsLoaded(page);
    await focusFirstRun(page);
  });

  test("sidebar metric families", async ({ page }) => {
    await page.screenshot({
      path: path.join(screenshotDir, "sidebar-metric-families.png"),
      fullPage: true,
    });
  });

  test("multipanel update_rank mlp (Llama fixture)", async ({ page }) => {
    await addSection(page);
    await addPanel(page, "Multipanel");
    await openPanelSettings(page, "Multipanel");
    const multipanel = panelByTitle(page, "Multipanel");
    await configureMultipanel(multipanel, "update_rank", String.raw`model\.layers\.\d+\.mlp`);
    await multipanel.locator(".multipanel-host").screenshot({
      path: path.join(screenshotDir, "multipanel-mlp-host.png"),
    });
  });

  test("multipanel update_rank self_attn (Llama fixture)", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await addSection(page);
    await addPanel(page, "Multipanel");
    await openPanelSettings(page, "Multipanel");
    const multipanel = panelByTitle(page, "Multipanel");
    await configureMultipanel(multipanel, "update_rank", LLAMA_SELF_ATTN_PATH_REGEX);
    await page.waitForTimeout(500);
    await multipanel.locator(".multipanel-host").screenshot({
      path: path.join(screenshotDir, "multipanel-self-attn-host.png"),
    });
    const firstGroup = multipanel.locator(".multipanel-group").first();
    if (await firstGroup.isVisible()) {
      await firstGroup.screenshot({
        path: path.join(screenshotDir, "multipanel-self-attn-group0.png"),
      });
    }
  });

  test("multipanel layout settle (3 captures after resize)", async ({ page }) => {
    await addSection(page);
    await addPanel(page, "Multipanel");
    await openPanelSettings(page, "Multipanel");
    const multipanel = panelByTitle(page, "Multipanel");
    await configureMultipanel(multipanel, "update_rank", String.raw`model\.layers\.\d+\.mlp`);
    for (const width of [900, 1100, 700]) {
      await page.setViewportSize({ width, height: 800 });
      await page.waitForTimeout(300);
      await multipanel.locator(".multipanel-host").screenshot({
        path: path.join(screenshotDir, `multipanel-mlp-w${width}.png`),
      });
    }
  });
});
