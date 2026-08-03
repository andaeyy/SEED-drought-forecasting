import { expect, test, type Page } from "@playwright/test";

function mockLayer(layer: string) {
  const lonEdges = [-106.8, -102.9, -99.0, -95.1];
  const latEdges = [26.1, 32.0, 37.9, 43.7];
  const features = [];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      const category = (row * 3 + col) % 6;
      const scalar = layer === "et" ? 0.45 + row * 0.8 + col * 0.35 : 0.11 + row * 0.07 + col * 0.025;
      features.push({
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [[
            [lonEdges[col], latEdges[row]],
            [lonEdges[col + 1], latEdges[row]],
            [lonEdges[col + 1], latEdges[row + 1]],
            [lonEdges[col], latEdges[row + 1]],
            [lonEdges[col], latEdges[row]]
          ]]
        },
        properties: layer === "drought"
          ? {
              row,
              col,
              pdry: 0.08 + category * 0.16,
              pdry_pct: 8 + category * 16,
              category,
              category_label: category === 0 ? "None" : `D${category - 1}`,
              risk_label: ["Normal", "Abnormally Dry", "Moderate Drought", "Severe Drought", "Extreme Drought", "Exceptional Drought"][category]
            }
          : { row, col, layer, value: scalar, units: layer === "et" ? "mm/day" : "m3/m3" }
      });
    }
  }
  return { type: "FeatureCollection", features };
}

async function mockCompletedForecast(page: Page) {
  const mockJobId = "8b9f1f2c-4a65-4f7a-bcf1-72d5c429618a";
  await page.route(/\/api\/forecast-jobs$/, async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ job_id: mockJobId, status: "queued" }) });
      return;
    }
    await route.fallback();
  });
  await page.route(new RegExp(`/api/forecast-jobs/${mockJobId}$`), async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        job_id: mockJobId,
        status: "complete",
        forecast: {
          target_day: "2026-08-05",
          horizon_days: 7,
          bounds: { lat_min: 26.1, lat_max: 43.7, lon_min: -106.8, lon_max: -95.1 },
          debug: {}
        }
      })
    });
  });
  await page.route(new RegExp(`/api/forecast-jobs/${mockJobId}/layers/(et|sm|drought)\\.geojson$`), async (route) => {
    const match = route.request().url().match(/\/layers\/(et|sm|drought)\.geojson$/);
    await route.fulfill({ contentType: "application/geo+json", body: JSON.stringify(mockLayer(match?.[1] ?? "drought")) });
  });
  await page.route(new RegExp(`/api/forecast-jobs/${mockJobId}/point-risk`), async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        requested_lat: 38.5,
        requested_lon: -99.5,
        grid_lat: 38.5,
        grid_lon: -99.5,
        grid_row: 1,
        grid_col: 1,
        pdry: 0.74,
        pdry_pct: 74,
        category: 1,
        category_label: "D0",
        risk_label: "Abnormally Dry",
        et_mm_per_day: 1.42,
        sm_m3_per_m3: 0.218
      })
    });
  });
}

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    content: document.documentElement.scrollWidth
  }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport);
}

async function waitForMap(page: Page) {
  await expect(page.locator('[data-map-ready="true"]')).toBeVisible({ timeout: 15_000 });
  await page.waitForTimeout(800);
}

test("landing page is complete and responsive", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Great Plains Drought Outlooks" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Open forecast dashboard/ })).toBeVisible();
  await expect(page.locator('img[alt="Aerial view of cultivated fields across the Great Plains"]')).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await page.screenshot({
    path: "../validation/screenshots/landing-desktop.png",
    fullPage: true
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload();
  await expect(page.getByRole("heading", { name: "Great Plains Drought Outlooks" })).toBeVisible();
  await expect.poll(() => page.locator('img[alt="Aerial view of cultivated fields across the Great Plains"]').evaluate(
    (image) => image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0
  )).toBe(true);
  await page.locator('img[alt="Aerial view of cultivated fields across the Great Plains"]').evaluate(
    (image) => image instanceof HTMLImageElement ? image.decode() : Promise.resolve()
  );
  await expectNoHorizontalOverflow(page);
  await page.screenshot({
    path: "../validation/screenshots/landing-mobile.png"
  });
});

test("dashboard renders selected model provenance for all horizons", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1600, height: 1000 });
  await mockCompletedForecast(page);
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Great Plains forecast workspace" })).toBeVisible();
  await waitForMap(page);
  await expect(page.getByRole("tab", { name: "ET" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "SM" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Dryness" })).toBeVisible();
  await expect(page.getByText("001_base_7var_weekly_ARconvlstm_7d_low_lr_clip")).toHaveCount(1);
  await expect(page.getByText("007_base_7var_weekly_DEconvlstm_7d_low_lr_wd_hidden48")).toHaveCount(1);

  await page.getByRole("button", { name: /Monthly/ }).click();
  await expect(page.getByText("029_base_7var_monthly_Seq2seqconvlstm_30d_compact_1layer")).toHaveCount(1);
  await expect(page.getByText("024_base_7var_monthly_DEconvlstm_30d_compact_1layer")).toHaveCount(1);

  await page.getByRole("button", { name: /Seasonal/ }).click();
  await expect(page.getByText("034_base_7var_seasonal_ARconvlstm_90d_compact_1layer")).toHaveCount(1);
  await expect(page.getByText("037_base_7var_seasonal_DEconvlstm_90d_low_lr_wd_hidden48")).toHaveCount(1);

  await page.getByRole("button", { name: /Weekly/ }).click();
  await expectNoHorizontalOverflow(page);
  const desktopHeight = await page.evaluate(() => ({ viewport: window.innerHeight, content: document.documentElement.scrollHeight }));
  expect(desktopHeight.content).toBeLessThanOrEqual(desktopHeight.viewport + 2);
  await page.screenshot({
    path: "../validation/screenshots/dashboard-desktop.png",
    fullPage: true
  });

  await page.getByRole("button", { name: /Run Weekly forecast/ }).click();
  await expect(page.locator('[role="status"]:visible').filter({ hasText: "Complete" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Download ET/SM dryness category GeoJSON" })).toBeEnabled();
  await page.getByRole("tab", { name: "ET" }).click();
  await expect(page.getByRole("button", { name: "Download Evapotranspiration endpoint GeoJSON" })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "Evapotranspiration" })).toBeVisible();
  await page.getByRole("tab", { name: "SM" }).click();
  await expect(page.getByRole("button", { name: "Download Soil-moisture endpoint GeoJSON" })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "Soil moisture" })).toBeVisible();
  await page.getByRole("tab", { name: "Dryness" }).click();
  await expect(page.getByRole("button", { name: "Download ET/SM dryness category GeoJSON" })).toBeEnabled();
  await page.getByLabel("Latitude").first().fill("38.5");
  await page.getByLabel("Longitude").first().fill("-99.5");
  await page.getByRole("button", { name: "Check coordinate" }).click();
  await expect(page.getByText("1.420 mm/day")).toBeVisible();
  await page.screenshot({
    path: "../validation/screenshots/dashboard-complete-desktop.png",
    fullPage: true
  });
  expect(pageErrors).toEqual([]);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Great Plains forecast workspace" })).toBeVisible();
  await waitForMap(page);
  await expectNoHorizontalOverflow(page);
  await page.screenshot({
    path: "../validation/screenshots/dashboard-mobile.png",
    fullPage: true
  });

  await page.getByRole("button", { name: /Run Weekly forecast/ }).click();
  await expect(page.locator('[role="status"]:visible').filter({ hasText: "Complete" })).toBeVisible();
  await page.getByLabel("Latitude").last().fill("38.5");
  await page.getByLabel("Longitude").last().fill("-99.5");
  await page.getByRole("button", { name: "Check coordinate" }).click();
  await expect(page.getByText("D0 - Abnormally Dry")).toBeVisible();
  await page.screenshot({
    path: "../validation/screenshots/dashboard-complete-mobile.png",
    fullPage: true
  });
});
