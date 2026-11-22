// Runs a painting page (maalaus.html, pesa.html, piirros.html or maalaus_turva.html) in headless Chromium,
// waits until the page title is "valmis" (done) and saves the canvas as a PNG.
// Usage: node aja_piirros.cjs <url> <ulos.png>   (ulos = output)
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [url, ulos] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1200, height: 700 } });
  p.on('console', m => console.log('selain:', m.text()));
  p.on('pageerror', e => console.log('virhe sivulla:', e.message));
  const alku = Date.now();
  await p.goto(url);
  await p.waitForFunction(() => document.title === 'valmis' || document.title.startsWith('virhe'), null, { timeout: 900000, polling: 1000 });
  const t = await p.title();
  if (t !== 'valmis') { console.error(t); process.exit(1); }
  const data = await p.evaluate(() => document.getElementById('c').toDataURL('image/png'));
  fs.writeFileSync(ulos, Buffer.from(data.split(',')[1], 'base64'));
  console.log('tallennettu', ulos, ((Date.now() - alku) / 1000).toFixed(1) + ' s');
  await b.close();
})();
