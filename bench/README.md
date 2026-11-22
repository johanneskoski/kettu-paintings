# Bench cover: how to run

A snowy park bench at night, drawn stroke by stroke. A small 3D render is used only as an underdrawing.

Files:
- `kansi.py`: small numpy ray tracer of the scene. With `opas=...` it writes the underdrawing data instead of a finished render.
- `opas.py`: turns the underdrawing into guide images (color, region, light, stroke direction, importance, form light).
- `piirros.html`: draws the picture on a 2400 x 1350 canvas with pastel and colored-pencil strokes.
- `aja_piirros.cjs`: opens the page in headless Chromium and saves the canvas as a PNG.

Steps (run in this folder, so the guide files land next to `piirros.html`):
1. `python3 kansi.py opas=./opas.npz ss=1` writes `opas.npz` and `opas.json`. Keep the default size (L=2400, K=1350).
2. `python3 opas.py ./opas.npz` writes `opas_vari.png`, `opas_data.png`, `opas_tarkeys.png`, `opas_muoto.png`, `opas_tiedot.json`.
   The path must contain a `/`, because the script takes the output folder from it.
3. Serve the folder over HTTP, for example `python3 -m http.server 8000`. The page loads its guide files with `fetch`.
4. `node aja_piirros.cjs http://localhost:8000/piirros.html bench.png`

Inputs: none. The scene, camera, and lamp are defined in `kansi.py`.
Optional: `python3 kansi.py ulos=kansi.png` saves the plain 3D render (other arguments: `siemen`, `fov`, `lamppu`, `teho`, `kuoppa`, `hiutaleet`, `valotus`).
Page URL parameters: `siemen` (random seed), `paperi` (paper color), `n1`..`n4` (stroke counts per stage), `hiutaleet` (snowflakes).

Needs: Python 3 with numpy, Pillow, and scipy; Node.js with the `playwright` package and its Chromium.
To rebuild the published picture (`../images/bench.jpg`), use the settings it was made with: `python3 kansi.py ss=1 kuoppa=0.08 lamppu=0.6,2.9,0.8 sisa=11 ulko=36 opas=./opas.npz`, then step 2, and open `piirros.html?siemen=1115` in step 4.
Run time: about 100 seconds in all, most of it the 3D render; the drawing itself takes about 5 seconds (tested on 22 Nov 2026, and the result matched the published picture up to JPEG noise). The runner waits at most 900 seconds for the page.
The page sets its title to the current stage, then to `valmis` (done) or `virhe: ...` (error). The runner waits for these strings.
