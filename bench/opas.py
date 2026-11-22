#!/usr/bin/env python3
"""Guide images for the hand drawing, made from the underdrawing. Kettu, 15 Nov 2026.

Reads the opas.npz produced by kansi.py and writes into the same folder:
  opas_vari.png     colors of the 3D image (strokes take their color from here)
  opas_data.png     R = region (0 dark, 1 ground, 2 wood, 3 metal, 4 snow on the bench, 5 hollow, 6 paw print) * 40,
                    G = lamp light (0..255), B = stroke direction (0..pi -> 0..255)
  opas_tarkeys.png  how precisely a spot is drawn (0..255): smaller strokes and more layers
Usage: python3 opas.py /path/opas.npz
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

polku = sys.argv[1]
d = np.load(polku)
vari, alue, valo, painauma = d['vari'], d['alue'], d['valo'], d['painauma']
H, W = alue.shape
tiedot = json.load(open(polku[:-4] + '.json'))

lum = vari.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255

# the lamp light, blurred, so that the strokes on the ground curve around the ring of light
valo_s = ndimage.gaussian_filter(np.clip(valo, 0, 2), 18)
gy, gx = np.gradient(valo_s)
voima = np.hypot(gx, gy)
kulma_maa = np.arctan2(gy, gx) + np.pi / 2                    # along the contour line of the light
paino = 0.35 * np.clip(voima / (np.percentile(voima[alue == 1], 70) + 1e-9), 0, 1)
oletus = np.radians(8)


def sekoita_kulmat(a, b, t):
    """Blend of two directions (mod pi) by way of the doubled angle."""
    x = (1 - t) * np.cos(2 * a) + t * np.cos(2 * b)
    y = (1 - t) * np.sin(2 * a) + t * np.sin(2 * b)
    return np.arctan2(y, x) / 2


kulma = np.full((H, W), np.radians(34), np.float32)          # darkness: slanted hatching
kohina = ndimage.gaussian_filter(np.random.default_rng(5).standard_normal((H, W)), 40)
kohina /= np.abs(kohina).max() + 1e-9
kulma += np.radians(9) * kohina

maa = (alue == 1) | (alue == 6)
kohina2 = ndimage.gaussian_filter(np.random.default_rng(6).standard_normal((H, W)), 70)
kohina2 /= np.abs(kohina2).max() + 1e-9
pimea = np.clip(1 - valo_s / 0.5, 0, 1)                     # in the dark the stroke direction varies more
oletus_k = oletus + np.radians(32) * kohina2 * pimea
kulma[maa] = sekoita_kulmat(oletus_k[maa], kulma_maa[maa], paino[maa])


# bench directions: the direction of the slats in the image (the bench's x axis)
def kuvasuunta(a, b):
    return np.arctan2(b[1] - a[1], b[0] - a[0])


import importlib.util  # noqa: E402  (the camera of kansi.py, from the same folder)
spec = importlib.util.spec_from_file_location('kansi', __file__.replace('opas.py', 'kansi.py'))
sys_argv = sys.argv
sys.argv = ['kansi.py', f'L={W}', f'K={H}', 'ss=1']
kansi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kansi)
sys.argv = sys_argv
rima = kuvasuunta(kansi.projisoi([-0.9, 0.6, -0.25]), kansi.projisoi([0.9, 0.6, -0.25]))
for a in (2, 4):
    kulma[alue == a] = rima

# metal: structure tensor from the region mask -> strokes run lengthwise along thin parts
m = ndimage.gaussian_filter((alue == 3).astype(np.float32), 3)
my, mx = np.gradient(m)
Jxx = ndimage.gaussian_filter(mx * mx, 6); Jyy = ndimage.gaussian_filter(my * my, 6); Jxy = ndimage.gaussian_filter(mx * my, 6)
kulma_m = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
kulma[alue == 3] = kulma_m[alue == 3]

# hollow: strokes curve around the rim of the hollow
kp = np.array(tiedot['kuoppa'])
kx, ky = kp[:, 0].mean(), kp[:, 1].mean()
rx, ry = (kp[:, 0].max() - kp[:, 0].min()) / 2, (kp[:, 1].max() - kp[:, 1].min()) / 2
yy, xx = np.mgrid[0:H, 0:W]
kuoppa = alue == 5
kulma[kuoppa] = rima                                         # in the hollow, the same strokes as in the rest of the snow: the form comes from tone

# cross-section lines: lines parallel to the bench that dip at the hollow (like a draftsman's contour lines)
KP = kansi.KUOPPA
y_lumi = kansi.B1[kansi.ISTUINLUMI][1]
poikki = []
for zz in np.linspace(KP['z'] - 1.2 * KP['rz'], KP['z'] + 1.15 * KP['rz'], 19):
    xs = np.linspace(KP['x'] - 1.4 * KP['rx'], KP['x'] + 1.4 * KP['rx'], 70)
    hs = kansi.kuopan_korkeus(xs, np.full_like(xs, zz))
    viiva_ = []
    for x_, h_ in zip(xs, hs):
        px_, py_ = kansi.projisoi([x_, y_lumi + h_, zz])
        viiva_.append([round(float(px_), 2), round(float(py_), 2), round(float(np.clip(-h_ / KP['syv'], 0, 1)), 3)])
    poikki.append(viiva_)
tiedot['poikkiviivat'] = poikki[::2]

# form light (`muoto`): as a draftsman would, the hollow is shaded with a side light so that its form stands out whatever the lamp's direction
yy_, xx_ = np.mgrid[0:H, 0:W]
fx_ = (2 * (xx_ + 0.5) / W - 1) * np.tan(kansi.FOV / 2)
fy_ = (1 - 2 * (yy_ + 0.5) / H) * np.tan(kansi.FOV / 2) * H / W
D_ = kansi.f[None, None] + fx_[..., None] * kansi.r + fy_[..., None] * kansi.u
t_ = (y_lumi - kansi.CAM[1]) / D_[..., 1]
Xw = kansi.CAM[0] + D_[..., 0] * t_; Zw = kansi.CAM[2] + D_[..., 2] * t_
alue_kuopan_lahella = ((alue == 4) | (alue == 5)) & (np.hypot((Xw - KP['x']) / KP['rx'], (Zw - KP['z']) / KP['rz']) < 1.5)
muoto = np.zeros((H, W), np.float32)
xs_ = Xw[alue_kuopan_lahella]; zs_ = Zw[alue_kuopan_lahella]
e_ = 0.002
gx_ = (kansi.kuopan_korkeus(xs_ + e_, zs_) - kansi.kuopan_korkeus(xs_ - e_, zs_)) / (2 * e_)
gz_ = (kansi.kuopan_korkeus(xs_, zs_ + e_) - kansi.kuopan_korkeus(xs_, zs_ - e_)) / (2 * e_)
N_ = np.stack([-gx_, np.ones_like(gx_), -gz_], 1); N_ /= np.linalg.norm(N_, axis=1, keepdims=True)
Lm = np.array([0.85, 0.75, 0.55]); Lm /= np.linalg.norm(Lm)
muoto[alue_kuopan_lahella] = (N_ @ Lm) - Lm[1]                 # compared with flat snow: + lighter, - more in shadow
muoto = ndimage.gaussian_filter(muoto, 1.0)
Image.fromarray((np.clip(128 + muoto * 300, 0, 255)).astype(np.uint8)).convert('RGB').save(polku.rsplit('/', 1)[0] + '/opas_muoto.png')

kulma = np.mod(kulma, np.pi)

# importance: the hollow and the prints matter most, then the bench and the lit snow, the darkness least
reunat = ndimage.gaussian_gradient_magnitude(lum, 1.5)
tarkeys = np.where(alue == 0, 0.18, 0.35) + 0.45 * np.clip(valo_s / 1.0, 0, 1)
tarkeys[(alue >= 2) & (alue <= 4)] = np.maximum(tarkeys[(alue >= 2) & (alue <= 4)], 0.7)
tarkeys += 0.6 * np.clip(reunat / 0.08, 0, 1)
d_kuoppa = np.hypot((xx - kx) / rx, (yy - ky) / ry)
tarkeys = np.maximum(tarkeys, np.clip(1.6 - d_kuoppa * 0.7, 0, 1))
jaljet = np.zeros((H, W), np.float32)
for j in tiedot['jaljet']:
    jaljet = np.maximum(jaljet, np.exp(-(((xx - j['x']) ** 2 + (yy - j['y']) ** 2) / (2 * 28.0 ** 2))))
tarkeys = np.maximum(tarkeys, jaljet)
tarkeys = np.clip(ndimage.gaussian_filter(tarkeys, 4), 0, 1)

# outlines: places where the region changes (not the edges of the hollow or the prints; those are drawn separately)
def ero(a1, a2):
    yhd = {(4, 5), (5, 4), (1, 6), (6, 1)}
    e = (a1 != a2)
    for p_, q_ in yhd:
        e &= ~((a1 == p_) & (a2 == q_))
    return e
raja = np.zeros((H, W), bool)
raja[:, :-1] |= ero(alue[:, :-1], alue[:, 1:]); raja[:-1, :] |= ero(alue[:-1, :], alue[1:, :])
siluetti = raja & (ndimage.binary_dilation(alue <= 1, iterations=2) & ndimage.binary_dilation((alue >= 2) & (alue <= 5), iterations=2))
lum_s = ndimage.gaussian_filter(lum, 1.5)
ly, lx = np.gradient(lum_s)
Jxx = ndimage.gaussian_filter(lx * lx, 3); Jyy = ndimage.gaussian_filter(ly * ly, 3); Jxy = ndimage.gaussian_filter(lx * ly, 3)
tangentti = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
ys_, xs_ = np.nonzero(raja)
valinta = np.random.default_rng(9).random(len(xs_)) < 0.30
reunat_lista = [[int(x_), int(y_), round(float(tangentti[y_, x_]), 3), int(siluetti[y_, x_])]
                for x_, y_ in zip(xs_[valinta], ys_[valinta])]
tiedot['reunat'] = reunat_lista

kansio = polku.rsplit('/', 1)[0]
Image.fromarray(vari).save(f'{kansio}/opas_vari.png')
data = np.stack([alue.astype(np.uint8) * 40,
                 (np.clip(valo_s / 1.3, 0, 1) * 255).astype(np.uint8),
                 (kulma / np.pi * 255).astype(np.uint8)], -1)
Image.fromarray(data).save(f'{kansio}/opas_data.png')
Image.fromarray((tarkeys * 255).astype(np.uint8)).convert('RGB').save(f'{kansio}/opas_tarkeys.png')
tiedot['kuopan_keski'] = [float(kx), float(ky)]
tiedot['kuopan_sateet'] = [float(rx), float(ry)]
tiedot['rima_kulma'] = float(rima)
json.dump(tiedot, open(f'{kansio}/opas_tiedot.json', 'w'), indent=1)
print('ok', W, H, 'alueet', {int(a): int((alue == a).sum()) for a in np.unique(alue)})
