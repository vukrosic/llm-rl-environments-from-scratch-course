import { existsSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try {
    return require('playwright');
  } catch (error) {
    const dependencyRoot = process.env.CODEX_NODE_MODULES;
    if (dependencyRoot) return require(path.join(dependencyRoot, 'playwright'));
    throw new Error('Playwright is missing. Run `npm install` first.', { cause: error });
  }
}

function browserOptions() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean);
  const executablePath = candidates.find(existsSync);
  return executablePath ? { executablePath, headless: true } : { headless: true };
}

const { chromium } = loadPlaywright();

const root = path.dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch(browserOptions());
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(`${pathToFileURL(path.join(root, 'index.html')).href}?all=1#slide-1`);
await page.waitForLoadState('networkidle');
const count = await page.locator('section.slide').count();
const results = [];

for (let i = 1; i <= count; i += 1) {
  await page.evaluate((n) => { location.hash = `slide-${n}`; }, i);
  await page.waitForTimeout(25);
  results.push(await page.locator('section.slide.active').evaluate((slide, n) => {
    const offenders = [...slide.querySelectorAll('*')].filter((el) => {
      const r = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden') return false;
      if (r.width === 0 && r.height === 0) return false;
      if (el.classList.contains('title-layout') || el.classList.contains('jobs-title-layout') || el.classList.contains('road-cover')) return false;
      return r.right > 1908 || r.bottom > 1068 || r.left < 12 || r.top < 12;
    }).map((el) => ({ tag: el.tagName, className: el.className, text: (el.textContent || '').trim().slice(0, 80), rect: el.getBoundingClientRect().toJSON() }));
    return { slide: n, scrollWidth: slide.scrollWidth, scrollHeight: slide.scrollHeight, offenders };
  }, i));
}

await browser.close();
writeFileSync(path.join(root, 'layout-audit.json'), `${JSON.stringify(results, null, 2)}\n`);
const failed = results.filter((r) => r.scrollWidth > 1920 || r.scrollHeight > 1080 || r.offenders.length);
console.log(JSON.stringify({ slides: count, failed: failed.length, failures: failed }, null, 2));
if (failed.length) process.exitCode = 1;
