# Fox's den painting: how to run
A fox's den under the roots of an old oak, in cross-section, painted stroke by stroke in pastel style. No photograph is needed: the shapes are drawn in code. Tested on 22 Nov 2026 with these steps.

**Files**
- `opas_pesa.py` draws the scene as masks and writes guide images: color, region, light, stroke direction and level of detail. It also writes `opas_tiedot.json` (tracks, edge lines, positions of the objects) and two check images (`tarkistus_alueet.png`, `tarkistus_vari.jpg`).
- `pesa.html` loads the guide images from `opas/` and paints 2400 x 1350 px on a canvas in layers: underpainting, body, blending, colored pencil, details, objects, pen work, edge lines, tracks, light, stars and snowfall. It sets the page title to `valmis` (done) at the end.
- `aja_piirros.cjs` opens the page in headless Chromium, waits for `valmis` and saves the canvas as a PNG.

**Needs**: Python 3 with numpy, Pillow and scipy; Node.js with Playwright and its Chromium.

**Commands** (in this folder)
```
mkdir opas
python3 opas_pesa.py opas
python3 -m http.server 8000 &
node aja_piirros.cjs http://localhost:8000/pesa.html den.png
```
The output folder must be named `opas` and must exist; the script does not create it. The page uses `fetch`, so serve it over HTTP rather than opening it as a file.

**Inputs**: none besides the code. Random seeds are fixed, so the same seed should give the same guide images and strokes. Optional URL parameters: `siemen` (random seed, default 1115), `paperi` (paper color), `n1`..`n5` (stroke counts per layer), `hiutaleet` (number of snowflakes), `valo` (blend mode of the lantern light, default `overlay`; `color-dodge` is the other one tried).

**Run time**: about 35 seconds for the guide images and 7 seconds for the painting (tested 22 Nov 2026). The driver gives up after 900 s. The result differs slightly from `../images/den.jpg`, because Johannes repainted the weasel's door by hand in the published picture. The door is in the code too.

**Good to know**: all positions (oak, roots, den, fox, map, door, weasel, Saturn, tracks) are fixed coordinates in `opas_pesa.py`. A few map and pen-work coordinates are repeated in `pesa.html`, so change both if you move the map. The result is shown in `../images/den.jpg`.
