#!/usr/bin/env python3
"""Guide images for the cover of the safety essay. Kettu, 17 Nov 2026.

It is based on Johannes's photograph of his desk lamp (17 Nov).
Nothing is copied from the photo as such: the guide images tell the strokes only the color, region,
light, direction and importance. The paper has been redrawn larger, and a form is placed on it:
the model's declaration and signature, and an empty line for the inspector.

Writes into the folder <ulos>:
  opas_vari.png     colors (strokes take their color from here)
  opas_data.png     R = region * 40 (0 background, 1 floor, 2 chest of drawers, 3 lamp, 4 light, 5 paper),
                    G = light 0..255, B = stroke direction (0..pi -> 0..255)
  opas_tarkeys.png  precision 0..255
  opas_muoto.png    flat 128
  opas_muste.png    the form's text and signature in the plane of the paper (grayscale, 255 = ink)
  opas_tiedot.json  edge lines and the corners of the paper
Usage: python3 opas_lamppu.py lamppu.jpg <ulos> [fi|en]
"""
import sys, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

lahde, ulos = sys.argv[1], sys.argv[2]
KIELI = sys.argv[3] if len(sys.argv) > 3 else 'fi'
W, H = 2400, 1350
vari = np.asarray(Image.open(lahde).convert('RGB').resize((W, H), Image.LANCZOS)).astype(np.float32)
lum = vari @ np.array([0.299, 0.587, 0.114], np.float32) / 255
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def monikulmio(pisteet):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in pisteet], fill=255)
    return np.asarray(m) > 0


def ellipsi(cx, cy, rx, ry, kulma=0.0):
    c, s = np.cos(np.radians(kulma)), np.sin(np.radians(kulma))
    u = (xx - cx) * c + (yy - cy) * s; v = -(xx - cx) * s + (yy - cy) * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def putki(pisteet, leveys):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).line([tuple(p) for p in pisteet], fill=255, width=int(leveys), joint='curve')
    return np.asarray(m) > 0


# ---------------------------------------------------------------- paper: the old one removed, a new larger one placed in the light
VANHA = [(474, 1030), (900, 930), (1284, 1125), (810, 1275)]
PAPERI = {'vy': (346, 975), 'oy': (878, 850), 'oa': (1358, 1094), 'va': (766, 1281)}   # `vy` top left, `oy` top right, `oa` bottom right, `va` bottom left
vanha = ndimage.binary_dilation(monikulmio(VANHA), iterations=14)
uusi = monikulmio([PAPERI[k] for k in ('vy', 'oy', 'oa', 'va')])
# floor under the old paper: filled in from the surroundings, stretched along the boards
reika = (vanha & ~uusi).astype(np.uint8) * 255
kuva8 = np.clip(vari, 0, 255).astype(np.uint8)
kuva8 = cv2.inpaint(kuva8, reika, 9, cv2.INPAINT_TELEA)
vari = kuva8.astype(np.float32)
lum = vari @ np.array([0.299, 0.587, 0.114], np.float32) / 255

# ---------------------------------------------------------------- the parts of the lamp, traced by hand
VARJOSTIN = [(1255, 222), (1215, 232), (1200, 243), (1150, 300), (1100, 345), (1035, 395), (1060, 420), (1120, 445), (1200, 463),
             (1280, 475), (1360, 482), (1425, 482), (1424, 400), (1422, 300), (1416, 222)]
HYLSY = [(1330, 125), (1355, 114), (1395, 118), (1412, 140), (1410, 222), (1290, 222), (1300, 190)]
NIVEL = [(1400, 198), (1468, 200), (1476, 246), (1404, 250)]
KAULA = [(1468, 234), (1500, 240), (1550, 250), (1600, 272), (1645, 307), (1675, 342), (1695, 382), (1707, 425), (1714, 475),
         (1718, 530), (1718, 585), (1714, 630), (1707, 656)]       # measured again after v4: the neck ran 10–20 px too high and too far right
# handle: a tilted cylinder with a rounded lower end. Measured from the photo with a grid (after v4):
# the handle was earlier too narrow and too upright.
KAHVA_AKSELI = [(1708, 660), (1656, 905)]
def _kahva():
    (x0, y0), (x1, y1) = KAHVA_AKSELI
    ux, uy = x1 - x0, y1 - y0; L = np.hypot(ux, uy); ux, uy = ux / L, uy / L; nx, ny = uy, -ux   # n points to the right
    leveys = lambda t: 30 + 4 * np.sin(np.pi * min(1.0, t * 1.15))
    vasen = [(x0 + ux * L * t + nx * leveys(t), y0 + uy * L * t + ny * leveys(t)) for t in np.linspace(0, 1, 12)]
    oikea = [(x0 + ux * L * t - nx * leveys(t), y0 + uy * L * t - ny * leveys(t)) for t in np.linspace(0, 1, 12)]
    pohja = [(x1 + nx * 31 * np.cos(a) + ux * 32 * np.sin(a), y1 + ny * 31 * np.cos(a) + uy * 32 * np.sin(a)) for a in np.linspace(0, np.pi, 13)]
    return [tuple(map(float, p)) for p in vasen + pohja + oikea[::-1]]
KAHVA = _kahva()
PALLO = (1649, 972, 31, 28)
KAULUS = [(1641, 930), (1667, 930), (1664, 950), (1644, 950)]
JALKA = (1582, 1016, 184, 81)                 # the base is a low dome: its silhouette is an ellipse, not a cylinder
KATKAISIN = [(1495, 926), (1511, 926), (1512, 950), (1520, 952), (1520, 974), (1486, 974), (1486, 952), (1494, 950)]
JOHTO = [(1752, 1016), (1790, 1030), (1840, 1033), (1890, 1020), (1930, 995), (1960, 962), (1985, 928), (2005, 896), (2025, 866),
         (2048, 834), (2072, 806), (2100, 786), (2135, 768)]          # fades into the dark in front of the chest of drawers
AUKKO = (1230, 436, 200, 32, 12.6)          # opening of the shade: center, semi-axes, angle

lamppu = monikulmio(VARJOSTIN) | monikulmio(HYLSY) | monikulmio(NIVEL) | putki(KAULA, 16) | monikulmio(KAHVA)
lamppu |= ellipsi(*PALLO) | monikulmio(KAULUS) | monikulmio(KATKAISIN) | putki(JOHTO, 11)
jalka = ellipsi(*JALKA)
lamppu |= jalka
aukko = ellipsi(*AUKKO)

# ---------------------------------------------------------------- regions
LATTIA_RAJA = np.where(xx[0] < 1358, 495.0, 835.0)
alue = np.full((H, W), 1, np.uint8)                       # floor by default
tausta = yy < LATTIA_RAJA[None, :]
alue[tausta] = 0
alue[tausta & (xx >= 1358)] = 2                           # chest of drawers behind the lamp
alue[lamppu] = 3
alue[aukko] = 4
alue[uusi] = 5

# dome of the base: the ball, its shadow and the switch are removed from the color guide, because they are painted separately
# and sharp. The dome's color is evened out with a normalized blur inside the dome, so that the floor color does not bleed
# onto the edges and the small color variation in the photo does not make blotches on the dome.
PALLON_VARJO = (1716, 991, 42, 14, 4)          # measured: a dark band at x 1680–1760, y 980–1005, to the right of the ball
kupu_poisto = ellipsi(*PALLO) | ellipsi(*PALLON_VARJO) | ndimage.binary_dilation(monikulmio(KATKAISIN), iterations=3)
kupu_poisto = ndimage.binary_dilation(kupu_poisto, iterations=3) & jalka
k8 = cv2.inpaint(np.clip(vari, 0, 255).astype(np.uint8), kupu_poisto.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
kj = jalka.astype(np.float32)
tasattu = np.stack([ndimage.gaussian_filter(k8[..., c] * kj, 7) for c in range(3)], -1) / (ndimage.gaussian_filter(kj, 7)[..., None] + 1e-6)
vari[jalka] = tasattu[jalka]

# ---------------------------------------------------------------- light
L = AUKKO
etaisyys = np.hypot(xx - L[0], yy - L[1])
valo = np.clip(ndimage.gaussian_filter(lum, 10) * 1.5, 0, 1)
valo[aukko] = 1.0
valo = np.maximum(valo, np.clip(1 - np.hypot((xx - L[0]) / 1.0, (yy - L[1]) / 0.6) / 300, 0, 1) ** 2 * (yy < 520))
valo[alue == 5] = np.maximum(valo[alue == 5], 0.62)

# paper color: the edge toward the lamp is warmer and brighter, the farthest edge slightly in shadow
pv = np.array(PAPERI['oy'], np.float32); pa = np.array(PAPERI['va'], np.float32)
t = np.clip(((xx - pa[0]) * (pv[0] - pa[0]) + (yy - pa[1]) * (pv[1] - pa[1])) / np.sum((pv - pa) ** 2), 0, 1)
paperi_vari = np.stack([236 + 10 * t, 231 + 9 * t, 222 + 3 * t], -1)
vari[uusi] = paperi_vari[uusi]

# ---------------------------------------------------------------- stroke direction
kohina = ndimage.gaussian_filter(np.random.default_rng(17).standard_normal((H, W)), 60)
kohina /= np.abs(kohina).max() + 1e-9
gy, gx = np.gradient(ndimage.gaussian_filter(lum, 2.5))
jxx = ndimage.gaussian_filter(gx * gx, 14); jyy = ndimage.gaussian_filter(gy * gy, 14); jxy = ndimage.gaussian_filter(gx * gy, 14)
rakenne = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2     # parallel to the edge
kulma = rakenne.copy()
# floor: the boards are straight and meet, in perspective, at a vanishing point. The vanishing point is fitted from the
# local directions of the boards (a masked structure tensor), so that knots and the edges of objects do not bend the strokes.
lattia_w = ndimage.binary_erosion(alue == 1, iterations=20).astype(np.float32)
gy3, gx3 = np.gradient(ndimage.gaussian_filter(lum, 2.0))
ww = ndimage.gaussian_filter(lattia_w, 120) + 1e-6
txx = ndimage.gaussian_filter(gx3 * gx3 * lattia_w, 120) / ww; tyy = ndimage.gaussian_filter(gy3 * gy3 * lattia_w, 120) / ww
txy = ndimage.gaussian_filter(gx3 * gy3 * lattia_w, 120) / ww
paik = 0.5 * np.arctan2(2 * txy, txx - tyy) + np.pi / 2
koherenssi = np.sqrt((txx - tyy) ** 2 + 4 * txy ** 2) / (txx + tyy + 1e-9)
otos = np.argwhere(lattia_w > 0)[::997]
A = np.stack([np.sin(paik[otos[:, 0], otos[:, 1]]), -np.cos(paik[otos[:, 0], otos[:, 1]])], -1)
b = np.sin(paik[otos[:, 0], otos[:, 1]]) * otos[:, 1] - np.cos(paik[otos[:, 0], otos[:, 1]]) * otos[:, 0]
painot = koherenssi[otos[:, 0], otos[:, 1]][:, None]
(PAKO_X, PAKO_Y), *_ = np.linalg.lstsq(A * painot, b * painot[:, 0], rcond=None)
print('pakopiste', round(float(PAKO_X)), round(float(PAKO_Y)))
laudat = np.arctan2(yy - PAKO_Y, xx - PAKO_X)
kulma[alue == 1] = laudat[alue == 1]
kulma[alue == 0] = (np.radians(34) + np.radians(10) * kohina)[alue == 0]
kulma[alue == 2] = (np.radians(2) * kohina)[alue == 2]     # drawers of the chest: horizontal
varjostin = monikulmio(VARJOSTIN) | monikulmio(HYLSY)
sade = np.arctan2(yy - 160, xx - 1340)                      # strokes on the shade run from the socket cap (`hylsy`) toward the rim
kulma[varjostin] = sade[varjostin]
tangentti = np.arctan2(-(xx - JALKA[0]) * JALKA[3] ** 2, (yy - JALKA[1]) * JALKA[2] ** 2 + 1e-6) + np.pi / 2
kulma[jalka] = tangentti[jalka]
kulma[aukko] = np.radians(AUKKO[4])
paperin_suunta = np.arctan2(PAPERI['oa'][1] - PAPERI['va'][1], PAPERI['oa'][0] - PAPERI['va'][0])
kulma[uusi] = paperin_suunta
kulma = np.mod(kulma, np.pi)

# ---------------------------------------------------------------- importance
tark = np.full((H, W), 0.28, np.float32)
tark[alue == 0] = 0.14; tark[alue == 2] = 0.22
tark[alue == 3] = 0.85; tark[alue == 4] = 0.9; tark[alue == 5] = 0.95
tark = np.maximum(tark, 0.55 * np.clip(valo - 0.35, 0, 1) * (alue == 1))
tark = ndimage.gaussian_filter(tark, 5)

# ---------------------------------------------------------------- the form, in the plane of the paper
U, V, S = 297.0, 210.0, 12          # the sheet in millimeters, S pixels per millimeter
tasainen = Image.new('L', (int(U * S), int(V * S)), 0)
d = ImageDraw.Draw(tasainen)
fontti = '/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf'
TEKSTIT = {
    'fi': ('Vakuutan olevani turvallinen.', 'Malli', 'Tarkastaja'),
    'en': ('I declare that I am safe.', 'Model', 'Inspector'),
}
paa, nimi1, nimi2 = TEKSTIT[KIELI]
koko = 20 if KIELI == 'fi' else 24                 # the heading fills the width of the sheet, up to the margins
iso = ImageFont.truetype(fontti, int(koko * S)); pieni = ImageFont.truetype(fontti, int(11 * S))
d.text((28 * S, 42 * S), paa, font=iso, fill=255)
viiva = int(1.0 * S)
d.line([(28 * S, 150 * S), (128 * S, 150 * S)], fill=230, width=viiva)
d.line([(169 * S, 150 * S), (269 * S, 150 * S)], fill=230, width=viiva)
d.text((28 * S, 154 * S), nimi1, font=pieni, fill=255)
d.text((169 * S, 154 * S), nimi2, font=pieni, fill=255)
# signature: "Claude" in cursive, in a single stroke, with slanted handwriting and a final stroke under the name.
# The letters are control points (x, height above the baseline), and the curve is smoothed with a spline.
from scipy.interpolate import splprep, splev
y0, x = 144.0, 34.0
ohjaus = []
def kirjain(pisteet, leveys):
    global x
    for px, py in pisteet: ohjaus.append((x + px, y0 - py))
    x += leveys
kirjain([(9, 17), (6, 20), (2, 17), (0, 10), (1, 3), (5, 0), (9, 2)], 9)                        # C
kirjain([(1, 3), (4, 12), (6, 20), (5, 22), (3, 18), (3, 8), (4, 1), (6, 0)], 6)                  # l
kirjain([(6, 6), (4, 8), (1, 6), (0, 2), (2, 0), (5, 3), (6, 8), (6, 2), (7, 0), (8, 1)], 8)      # a
kirjain([(0, 7), (0, 2), (2, 0), (5, 2), (6, 7), (6, 2), (7, 0), (8, 1)], 8)                      # u
kirjain([(6, 6), (4, 8), (1, 6), (0, 2), (2, 0), (5, 3), (7, 12), (8, 21), (8, 12), (7, 2), (8, 0), (10, 1)], 10)   # d
kirjain([(1, 3), (5, 6), (4, 8), (1, 7), (0, 3), (2, 0), (5, 0), (8, 3)], 8)                      # e
kirjain([(4, 3), (6, 0), (3, -3), (-20, -4), (-44, -2)], 0)                                       # final stroke, as an underline
ohjaus = np.array(ohjaus)
tck, _ = splprep([ohjaus[:, 0], ohjaus[:, 1]], s=2.0, k=3)
kx, ky = splev(np.linspace(0, 1, 900), tck)
vino = 0.28                                           # slant of the handwriting to the right
pts = [(px + (y0 - py) * vino, py) for px, py in zip(kx, ky)]
d.line([(p[0] * S, p[1] * S) for p in pts], fill=255, width=int(0.85 * S), joint='curve')
# fold in the middle: a bright and a dark line side by side (saved separately to a file)
mallinen = np.asarray(tasainen).astype(np.float32) / 255

lahto = np.float32([[0, 0], [U * S, 0], [U * S, V * S], [0, V * S]])
kohde = np.float32([PAPERI['vy'], PAPERI['oy'], PAPERI['oa'], PAPERI['va']])
Hm = cv2.getPerspectiveTransform(lahto, kohde)
muste = cv2.warpPerspective(mallinen, Hm, (W, H), flags=cv2.INTER_AREA if False else cv2.INTER_LINEAR)
muste = np.clip(muste, 0, 1)

# ---------------------------------------------------------------- edge lines for the lamp and the chest of drawers
gm = np.hypot(gx, gy)
maski = ndimage.binary_dilation((alue == 3) | (alue == 2), iterations=2) & ~aukko & ~ndimage.binary_dilation(varjostin, iterations=4)
maski &= ~ndimage.binary_dilation(jalka, iterations=4)       # the dome of the base is smooth paint and its edge is soft: no sketch lines
raja_arvo = np.percentile(gm[maski], 82)
ehdokkaat = np.argwhere(maski & (gm > raja_arvo))
rng2 = np.random.default_rng(7)
valitut = ehdokkaat[rng2.choice(len(ehdokkaat), size=min(5000, len(ehdokkaat)), replace=False)]
reunat = [[int(x), int(y), float(np.arctan2(gy[y, x], gx[y, x]) + np.pi / 2), float(lum[y, x])] for y, x in valitut]
json.dump({'reunat': reunat, 'paperi': PAPERI, 'kaula': KAULA, 'johto': JOHTO, 'aukko': AUKKO, 'jalka': JALKA,
           'pallo': PALLO, 'pallon_varjo': PALLON_VARJO, 'katkaisin': KATKAISIN, 'kahva': KAHVA, 'kahva_akseli': KAHVA_AKSELI,
           'varjostin': VARJOSTIN, 'hylsy': HYLSY,
           'homografia': Hm.tolist(), 'mm': S}, open(f'{ulos}/opas_tiedot.json', 'w'))

data = np.stack([alue.astype(np.float32) * 40, valo * 255, kulma / np.pi * 255], -1)
Image.fromarray(np.clip(vari, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_vari.png')
Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_data.png')
Image.fromarray(np.clip(tark * 255, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_tarkeys.png')
Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(f'{ulos}/opas_muoto.png')
Image.fromarray(np.clip(muste * 255, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_muste.png')
paletti = np.array([[25, 20, 15], [170, 110, 50], [70, 40, 20], [120, 120, 90], [255, 245, 200], [240, 240, 235]], np.uint8)
tark_kuva = paletti[alue].copy(); tark_kuva[muste > 0.3] = (200, 30, 30)
Image.fromarray(tark_kuva).resize((W // 3, H // 3)).save(f'{ulos}/tarkistus_alueet.png')
# check of the tracings: the lamp's outline in cyan on top of the photograph (the image stays in the working folder, not in the repo)
paallys = np.asarray(Image.open(lahde).convert('RGB').resize((W, H), Image.LANCZOS)).copy()
raja = (alue == 3) & ~ndimage.binary_erosion(alue == 3, iterations=2)
paallys[raja] = (0, 255, 255)
Image.fromarray(paallys).crop((1000, 60, 2200, 1110)).save(f'{ulos}/tarkistus_lamppu.png')
print('alueet', np.round(np.bincount(alue.ravel(), minlength=6) / alue.size, 3))
