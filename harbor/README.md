# Harbor painting: how to run
A night harbor painted stroke by stroke in pastel style, from a photograph. Not run or tested for this note; the steps are read from the code.

**Files**
- `opas_kuvasta.py` reads the photo and writes guide images: color, region, light, stroke direction, level of detail, plus edge lines (`opas_tiedot.json`) and a check image (`tarkistus_alueet.png`).
- `maalaus.html` loads the guide images from `opas/` and paints 2400 x 1350 px on a canvas in layers: underpainting, body, blending, colored pencil, details, reflection, lamp, buoys, gouache. It sets the page title to `valmis` (done) at the end.
- `aja_piirros.cjs` opens the page in headless Chromium, waits for `valmis` and saves the canvas as a PNG.

**Needs**: Python 3 with numpy, Pillow and scipy; Node.js with Playwright and its Chromium.

**Commands** (in this folder)
```
mkdir opas
python3 opas_kuvasta.py satama.jpg opas
python3 -m http.server 8000 &
node aja_piirros.cjs http://localhost:8000/maalaus.html harbor.png
```
The output folder must be named `opas` and must exist; the script does not create it. The page uses `fetch`, so serve it over HTTP rather than opening it as a file. Optional URL parameters: `siemen` (random seed, default 1115), `paperi` (paper color), `n1`..`n4` (stroke counts per layer). Run time is not stated in the code. The driver prints the seconds it took and gives up after 900 s.

**The photo is not included.** The original is Johannes's evening photo of a harbor, and several parts are made for that one image:
- `opas_kuvasta.py`: the crop box `(0, 500, 1512, 1350)`, the shoreline points `ranta_x` / `ranta_y`, the hand-outlined boat polygons (`vene_maski`), the buoy ellipses `POIJUT`, the buoy poles `TANGOT`, and the brightness and saturation thresholds for sky, covers and reflection. The lamp is searched for as the brightest spot in the top 45 % of the image.
- `maalaus.html`: the fixed lamp position `LAMPPU`, the lamp post coordinates, `POIJUT`, `TANGOT`, the ropes `KOYDET`, the y limits (600, 760, 840) and the palette `LIIDUT`.

With another photo, replace these by hand. The result is shown in `../images/harbor.jpg`.
