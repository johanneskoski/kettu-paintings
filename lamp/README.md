# Lamp cover: how to run

A desk lamp on a wooden floor, lighting a signed form. Painted stroke by stroke in pastel, from guide images made from a photograph.

Files:
- `opas_lamppu.py`: makes the guide images from the photo, and draws the form text and the "Claude" signature onto the paper.
- `maalaus_turva.html`: paints the picture on a 2400 x 1350 canvas.
- `aja_piirros.cjs`: opens a page in headless Chromium and saves the canvas as a PNG. It runs any painting page given as a URL.

Steps (run in this folder):
1. `mkdir opas`, then `python3 opas_lamppu.py lamppu.jpg opas en` (or `fi` for the Finnish form). The script does not create the folder.
   It writes `opas_vari.png`, `opas_data.png`, `opas_tarkeys.png`, `opas_muoto.png`, `opas_muste.png`, `opas_tiedot.json`, and two check images (`tarkistus_*.png`).
2. Serve the folder over HTTP, for example `python3 -m http.server 8000`. The page loads its guide files with `fetch`.
3. `node aja_piirros.cjs "http://localhost:8000/maalaus_turva.html?opas=opas" lamp.png`

Input: one photograph. The original photo (by Johannes) is not included, so the picture cannot be reproduced exactly.
The code is made for that one photo. Everything below is in pixels of the photo after it is resized to 2400 x 1350:
- `opas_lamppu.py`: the old and new paper corners (`VANHA`, `PAPERI`), all hand-traced lamp parts (`VARJOSTIN`, `HYLSY`, `NIVEL`, `KAULA`, `KAHVA_AKSELI`, `PALLO`, `KAULUS`, `JALKA`, `KATKAISIN`, `JOHTO`, `AUKKO`, `PALLON_VARJO`), the floor line (`LATTIA_RAJA`), the shade stroke center, and the crop of the check image.
- `maalaus_turva.html`: stages 5, 6, and 9 use fixed pixel positions for the shade, joint, base, switch, handle, cord, and highlights, and `JALKA_PALETTI` is a palette measured from the photo.
With another photo you need to trace these again. Stages 1 to 4 follow the guide images and need no changes.

Page URL parameters: `opas` (guide folder), `siemen` (random seed), `paperi` (paper color), `muste` (ink color), `n1`..`n4` (stroke counts).
Needs: Python 3 with numpy, opencv-python (`cv2`), Pillow, and scipy; the Caladea font at `/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf`; Node.js with `playwright` and its Chromium.
Run time: not stated in the code. The runner waits at most 900 seconds and prints the seconds it took. The page title ends as `valmis` (done) or `virhe: ...` (error).
