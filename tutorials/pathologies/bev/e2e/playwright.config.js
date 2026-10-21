import { defineConfig, devices } from "@playwright/test";

const port = process.env.BEV_E2E_PORT ?? "18765";
const baseURL = `http://127.0.0.1:${port}`;

/** @type {import('@playwright/test').PlaywrightTestConfig} */
export default defineConfig({
  testDir: "./specs",
  testIgnore: process.env.BEV_CAPTURE_SCREENSHOTS ? [] : ["**/capture-screenshots.spec.js"],
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL,
    trace: "on-first-retry",
    ...devices["Desktop Chrome"],
  },
  webServer: {
    command: "bash ./serve.sh",
    url: `${baseURL}/api/runs`,
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
});
