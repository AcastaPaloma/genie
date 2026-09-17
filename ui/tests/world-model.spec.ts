import { test, expect } from '@playwright/test';

// Needs a running world model: python3 experiments/play_server.py --port 8018 --no-check, then
// WORLD_MODEL_URL=http://127.0.0.1:8018 npx playwright test tests/world-model.spec.ts
const world = process.env.WORLD_MODEL_URL ?? 'http://127.0.0.1:8008';

test('generated world-model frames reach the stimulus capture path and follow the keys', async ({ page, request }) => {
  let reachable = false;
  try { reachable = (await request.get(`${world}/frame`, { timeout: 3000 })).ok(); } catch {}
  test.skip(!reachable, `No play server at ${world}; start experiments/play_server.py first`);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/world-test.html');
  const app = page.locator('#world-test');
  await expect(app).toHaveAttribute('data-ready', 'true', { timeout: 30000 });
  await expect.poll(async () => Number(await app.getAttribute('data-captures')), { timeout: 30000 }).toBeGreaterThanOrEqual(5);
  expect(Number(await app.getAttribute('data-generated'))).toBeGreaterThanOrEqual(5);
  expect(Number(await app.getAttribute('data-maximum'))).toBeGreaterThan(0);
  await expect(app).toHaveAttribute('data-label', 'idle (W)');
  await page.keyboard.down('a');
  await expect(app).toHaveAttribute('data-label', 'A', { timeout: 10000 });
  await expect(app).toHaveAttribute('data-code', '3');
  await page.keyboard.up('a');
  await expect(app).toHaveAttribute('data-label', 'idle (W)', { timeout: 10000 });
  const before = Number(await app.getAttribute('data-captures'));
  await expect.poll(async () => Number(await app.getAttribute('data-captures')), { timeout: 10000 }).toBeGreaterThan(before);
  expect(await app.getAttribute('data-error')).toBeNull();
  expect(errors).toEqual([]);
});
