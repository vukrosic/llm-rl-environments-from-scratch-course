import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, rmSync } from 'node:fs';
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
const out = path.join(root, 'renders');
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const browser = await chromium.launch(browserOptions());
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(`${pathToFileURL(path.join(root, 'index.html')).href}?all=1#slide-1`);
await page.waitForLoadState('networkidle');
const count = await page.locator('section.slide').count();

for (let i = 1; i <= count; i += 1) {
  await page.evaluate((n) => { location.hash = `slide-${n}`; }, i);
  await page.waitForTimeout(90);
  await page.screenshot({ path: path.join(out, `slide-${String(i).padStart(2, '0')}.png`) });
}
await browser.close();

for (let group = 0; group < Math.ceil(count / 9); group += 1) {
  const start = group * 9 + 1;
  const end = Math.min(start + 8, count);
  const files = [];
  for (let i = start; i <= end; i += 1) files.push(path.join(out, `slide-${String(i).padStart(2, '0')}.png`));
  try {
    execFileSync('magick', ['montage', ...files, '-thumbnail', '640x360', '-tile', '3x3', '-geometry', '+0+0', path.join(out, `contact-${group + 1}.png`)]);
  } catch {
    console.warn('ImageMagick is unavailable; slide PNGs were rendered without contact sheets.');
    break;
  }
}

console.log(`Rendered ${count} slides to ${out}`);
