import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.ts',
  timeout: 60000,
  expect: { timeout: 20000 },
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:5173', headless: true, launchOptions: { args: ['--enable-webgl', '--ignore-gpu-blocklist', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] } },
  webServer: { command: 'npm run dev -- --port 5173 --strictPort', url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI },
});
