#!/usr/bin/env python3
"""Guide images for a hand-drawn picture, made from a photograph. Kettu, 16 Nov 2026.

The source is Johannes's photo of a harbor from the evening of 16 Nov. Nothing is copied from the photo as such:
the guide images only tell the strokes their color, region, light, direction and importance.

Writes into the folder <ulos> (output):
  opas_vari.png     the colors of the photo (the strokes take their color from this)
  opas_data.png     R = region * 40 (0 sky, 1 water, 2 land and trees, 3 covers and boats, 4 light, 5 buoys),
                    G = light 0..255, B = stroke direction (0..pi -> 0..255)
  opas_tarkeys.png  level of detail 0..255
  opas_muoto.png    flat 128 (form light is not used)
Usage: python3 opas_kuvasta.py satama.jpg <ulos>
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

lahde, ulos = sys.argv[1], sys.argv[2]
W, H = 2400, 1350
kuva = Image.open(lahde).convert('RGB').crop((0, 500, 1512, 1350)).resize((W, H), Image.LANCZOS)
vari = np.asarray(kuva).astype(np.float32)
lum = vari @ np.array([0.299, 0.587, 0.114], np.float32) / 255
r, g, b = vari[..., 0], vari[..., 1], vari[..., 2]
mx, mn = vari.max(-1), vari.min(-1)
sat = (mx - mn) / (mx + 1e-6)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# shoreline (x, y) in full-image coordinates: below it is water
ranta_x = np.array([0, 300, 520, 700, 900, 1150, 1400, 1700, 2000, 2400], np.float32)
ranta_y = np.array([760, 760, 745, 700, 655, 640, 610, 580, 570, 565], np.float32)
ranta = np.interp(xx[0], ranta_x, ranta_y)
vesi = yy > ranta[None, :]

# lamp: the brightest spot in the upper part
yla = lum.copy(); yla[int(H * 0.45):] = 0
ly, lx = np.unravel_index(np.argmax(ndimage.gaussian_filter(yla, 6)), yla.shape)
etaisyys = np.hypot(xx - lx, yy - ly)

alue = np.full((H, W), 2, np.uint8)                       # land and trees by default
taivas = (~vesi) & (lum < 0.13) & (yy < ranta[None, :] - 60)
taivas &= ndimage.binary_opening(taivas, iterations=2)
alue[taivas] = 0
peite = (~vesi) & (lum > 0.33) & (sat < 0.30)
peite = ndimage.binary_opening(peite, iterations=2)
alue[peite] = 3
alue[vesi] = 1
# boats on the water: hand-outlined areas where brightness separates the boat from the water
from PIL import ImageDraw
vene_maski = Image.new('L', (W, H), 0); dv = ImageDraw.Draw(vene_maski)
dv.polygon([(0, 600), (140, 595), (180, 670), (470, 680), (480, 830), (300, 840), (140, 870), (0, 870)], fill=255)
dv.polygon([(628, 665), (885, 665), (885, 765), (628, 765)], fill=255)
dv.polygon([(1295, 540), (1515, 540), (1515, 655), (1295, 655)], fill=255)
dv.polygon([(2250, 480), (2400, 480), (2400, 560), (2250, 560)], fill=255)
vene_alue = (np.asarray(vene_maski) > 0) & (lum > 0.13)
alue[ndimage.binary_opening(vene_alue, iterations=1)] = 3
# buoys: hand-marked ellipses (center, radii)
POIJUT = [(330, 895, 36, 28), (268, 932, 30, 22), (503, 866, 26, 20), (905, 820, 36, 23), (1010, 760, 21, 15),
          (990, 798, 22, 12), (940, 1203, 55, 38), (1480, 692, 11, 8), (1488, 708, 17, 12), (1800, 648, 14, 7),
          (1922, 680, 16, 9), (1947, 670, 14, 9), (2158, 618, 15, 10), (2172, 633, 18, 13), (2026, 912, 23, 16)]
for (cx, cy, rx, ry) in POIJUT:
    alue[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 5
# light: the lamp and its reflection on the water
heijastus = vesi & (lum > 0.62) & (np.abs(xx - lx) < 260)
heijastus = ndimage.binary_dilation(heijastus, iterations=2)
alue[heijastus] = 4
alue[etaisyys < 34] = 4
# the buoy poles go into the same region, so that the water strokes do not pick up the pale color of a pole
TANGOT = [(930, 1168, 925, 1003), (2026, 896, 2026, 846), (1922, 671, 1922, 640), (1947, 661, 1948, 628), (2158, 608, 2158, 588),
          (1800, 646, 1760, 632), (895, 800, 890, 705), (985, 748, 985, 705)]
tanko_maski = Image.new('L', (W, H), 0); dt = ImageDraw.Draw(tanko_maski)
for (x1, y1, x2, y2) in TANGOT:
    pitka = np.hypot(x2 - x1, y2 - y1) > 120
    dt.line([(x1, y1), (x2, y2)], fill=255, width=14 if pitka else 9)
    if pitka: dt.line([(x2 - 3, y2 - 40), (x2 + 16, y2 - 16), (x2 + 2, y2 + 4), (x2 - 16, y2 - 19), (x2 - 3, y2 - 40)], fill=255, width=8)   # ring: outline only, the water shows through
alue[np.asarray(tanko_maski) > 0] = 5                     # the poles only after the reflection, so that the light does not cover them

# light intensity: the glow of the lamp and the brightness of the reflection
valo = np.clip(ndimage.gaussian_filter(lum, 10) * 1.6, 0, 1)
valo = np.maximum(valo, np.clip(1 - etaisyys / 260, 0, 1) ** 2)

# stroke direction
kohina = ndimage.gaussian_filter(np.random.default_rng(16).standard_normal((H, W)), 60)
kohina /= np.abs(kohina).max() + 1e-9
kulma = np.zeros((H, W), np.float32)
# structure tensor for the shapes in the photo (covers, boats, trees)
gy, gx = np.gradient(ndimage.gaussian_filter(lum, 2.5))
jxx = ndimage.gaussian_filter(gx * gx, 12); jyy = ndimage.gaussian_filter(gy * gy, 12); jxy = ndimage.gaussian_filter(gx * gy, 12)
rakenne = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2      # parallel to the edge
kulma[:] = rakenne
# trees vertical, but the ground at the shore horizontal: near the shoreline the strokes run along the surface of the ground
rannasta = ranta[None, :] - yy
maahan = np.clip((rannasta - 140) / 120, 0, 1)                             # 0 at the shore, 1 at the height of the trees
pysty = np.pi / 2 + np.radians(8) * kohina
vaaka = np.radians(6) * kohina
kulma[alue == 2] = np.where(maahan > 0.5, pysty, vaaka)[alue == 2]
taivas_kulma = np.radians(34) + np.radians(10) * kohina
sateittain = np.arctan2(yy - ly, xx - lx) % np.pi                           # radially around the lamp
hehku = np.clip(1 - etaisyys / 180, 0, 1)
kulma[alue == 0] = (taivas_kulma * (1 - hehku) + sateittain * hehku)[alue == 0]
kulma[alue == 1] = (np.radians(4) * kohina)[alue == 1]                      # water horizontal
kulma[alue == 4] = np.where(etaisyys < 80, sateittain, np.radians(2) * kohina)[alue == 4]
kulma[alue == 5] = 0
vene_vedessa = (np.asarray(vene_maski) > 0) & (alue == 3)
kulma = np.mod(kulma, np.pi)

# importance: lamp, reflection, buoys and boats in more detail
tark = np.full((H, W), 0.25, np.float32)
tark[alue == 3] = 0.7; tark[alue == 5] = 0.95; tark[alue == 4] = 0.9
tark = np.maximum(tark, np.clip(1 - etaisyys / 420, 0, 1))
tark[vene_vedessa] = 0.92                                                   # the boats are drawn with small strokes
tark = ndimage.gaussian_filter(tark, 6)

# edge lines for the boats and covers: the strongest brightness edges, direction along the edge
import json
gm = np.hypot(gx, gy)
maski = ndimage.binary_dilation(alue == 3, iterations=3)
raja_arvo = np.percentile(gm[maski], 80)
ehdokkaat = np.argwhere(maski & (gm > raja_arvo))
rng = np.random.default_rng(7)
valitut = ehdokkaat[rng.choice(len(ehdokkaat), size=min(6000, len(ehdokkaat)), replace=False)]
reunat = [[int(x), int(y), float(np.arctan2(gy[y, x], gx[y, x]) + np.pi / 2), float(lum[y, x])] for y, x in valitut]
json.dump({'reunat': reunat}, open(f'{ulos}/opas_tiedot.json', 'w'))

data = np.stack([alue.astype(np.float32) * 40, valo * 255, kulma / np.pi * 255], -1)
Image.fromarray(vari.astype(np.uint8)).save(f'{ulos}/opas_vari.png')
Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_data.png')
Image.fromarray(np.clip(tark * 255, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_tarkeys.png')
Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(f'{ulos}/opas_muoto.png')
# check image of the regions
paletti = np.array([[20, 20, 40], [90, 110, 60], [40, 60, 30], [200, 200, 210], [255, 240, 200], [230, 110, 60]], np.uint8)
Image.fromarray(paletti[alue]).resize((W // 4, H // 4)).save(f'{ulos}/tarkistus_alueet.png')
print('lamppu', lx, ly, 'alueet', np.bincount(alue.ravel(), minlength=6) / alue.size)
