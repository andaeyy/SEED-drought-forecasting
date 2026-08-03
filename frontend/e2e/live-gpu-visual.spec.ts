import { expect, test } from "@playwright/test";

test.skip(process.env.RUN_LIVE_SEED_E2E !== "1", "Requires an active GPU API on localhost:8000");

async function waitForLiveForecast(page: import("@playwright/test").Page) {
  await expect(page.locator('[data-map-ready="true"]')).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: /Run Weekly forecast/ }).click();
  await expect(page.locator('[role="status"]:visible').filter({ hasText: "Complete" })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole("button", { name: "Download ET/SM dryness category GeoJSON" })).toBeEnabled({ timeout: 30_000 });
  await page.waitForTimeout(3_000);
}

test("live A40 weekly forecast renders the full Great Plains grid", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/dashboard");
  await waitForLiveForecast(page);
  await page.getByLabel("Latitude").first().fill("38.5");
  await page.getByLabel("Longitude").first().fill("-99.5");
  await page.getByRole("button", { name: "Check coordinate" }).click();
  await expect(page.getByText("Grid: 38.438, -99.563")).toBeVisible();
  await page.screenshot({
    path: "../validation/screenshots/dashboard-live-complete-desktop.png",
    fullPage: true
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/dashboard");
  await waitForLiveForecast(page);
  await page.getByLabel("Latitude").last().fill("38.5");
  await page.getByLabel("Longitude").last().fill("-99.5");
  await page.getByRole("button", { name: "Check coordinate" }).click();
  await expect(page.getByText("Grid: 38.438, -99.563")).toBeVisible();
  await page.screenshot({
    path: "../validation/screenshots/dashboard-live-complete-mobile.png",
    fullPage: true
  });

  expect(pageErrors).toEqual([]);
});
