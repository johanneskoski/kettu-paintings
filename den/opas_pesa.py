#!/usr/bin/env python3
"""A fox's den under the roots of an old oak: guide images without a photograph. Kettu, 16 Nov 2026.

Johannes's idea, 16 Nov: a warm den in the middle of a Finnish forest in autumn or winter,
under the roots of an old oak. The picture is a cross-section: you can see inside the fox's den, because this memory is visible.
Of the weasel's den, only the door at the end of the tunnel is shown, because the weasel's inner world is the weasel's own.

Writes into the folder <ulos> (output) the same guide images as opas_kuvasta.py:
  opas_vari.png, opas_data.png (R = region * 40, G = light, B = direction), opas_tarkeys.png, opas_muoto.png
  and opas_tiedot.json (tracks, edge lines and the positions of the objects).
Regions: 0 sky, 1 snow, 2 earth, 3 wood and roots (also the spruces), 4 warm light (den and tunnels), 5 small objects.
Usage: python3 opas_pesa.py opas
"""
import sys, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ulos = sys.argv[1]
W, H = 2400, 1350
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
X = xx[0]


def kohina(solu, siemen):
    r = np.random.default_rng(siemen)
    g = r.standard_normal((H // solu + 4, W // solu + 4)).astype(np.float32)
    k = ndimage.zoom(g, solu, order=3)[:H, :W]
    return k / (np.abs(k).max() + 1e-9)


n1, n2, n3 = kohina(40, 1), kohina(9, 2), kohina(120, 3)
vari = np.zeros((H, W, 3), np.float32)
alue = np.zeros((H, W), np.uint8)
kulma = np.zeros((H, W), np.float32)
valo = np.zeros((H, W), np.float32)
tark = np.full((H, W), 0.25, np.float32)


def maski(piirto):
    m = Image.new('L', (W, H), 0)
    piirto(ImageDraw.Draw(m))
    return np.asarray(m) > 127


def aseta(m, rgb, a=None, k=None, t=None, v=None):
    rgb = np.asarray(rgb, np.float32)
    vari[m] = rgb if rgb.ndim == 1 else rgb[m]
    if a is not None: alue[m] = a
    if k is not None: kulma[m] = k if np.isscalar(k) else k[m]
    if t is not None: tark[m] = np.maximum(tark[m], t)
    if v is not None: valo[m] = np.maximum(valo[m], v if np.isscalar(v) else v[m])


def sekoita(a, b, t):
    t = np.clip(t, 0, 1)[..., None]
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


# ------------------------------------------------------------------ landscape: sky, snow plane, cross-section of snow and earth
taka = 520 + 14 * np.sin(X / 210) + 8 * np.sin(X / 67 + 1)            # base of the background forest in the snow
etu = 652 + 10 * np.sin(X / 300 + 0.5) + 5 * np.sin(X / 90)             # front edge of the snow plane, i.e. the top edge of the cross-section
lumi_ala = etu + 58 + 8 * np.sin(X / 150 + 2)                             # bottom edge of the snow layer in the cross-section
TAKA, ETU, LALA = taka[None, :], etu[None, :], lumi_ala[None, :]

m = yy < TAKA                                                             # sky: deep blue, lighter at the horizon
t = yy / 520
aseta(m, sekoita([14, 19, 46], [44, 58, 112], t ** 1.6) + n3[..., None] * 6, 0, np.radians(34) + np.radians(10) * n1, v=0.05)
m = (yy >= TAKA) & (yy < ETU)                                             # snow plane in perspective: bluish at the back, lighter at the front
syv = (yy - TAKA) / (ETU - TAKA)
aseta(m, sekoita([112, 128, 172], [196, 205, 230], syv) + n2[..., None] * 5, 1, np.radians(2) * n1, v=0.25 + 0.2 * syv)
m = (yy >= ETU) & (yy < LALA)                                             # cross-section of the snow: layers
kerros = 0.5 + 0.5 * np.sin((yy - ETU) / 7 + n1 * 2)
aseta(m, sekoita([208, 214, 234], [150, 160, 198], (yy - ETU) / 60) - kerros[..., None] * 10, 1, np.radians(1.5) * n1, v=0.22)
m = yy >= LALA                                                            # earth: layered, darkens downwards
syv = np.clip((yy - LALA) / (H - LALA), 0, 1)
juova = np.sin((yy - LALA) / 23 + n1 * 2.4) * 0.5 + 0.5
maa_vari = sekoita([78, 56, 40], [34, 24, 18], syv ** 0.8) * (0.86 + 0.14 * juova[..., None]) + n2[..., None] * 6
maa_kulma = np.radians(4) * np.sin(xx / 140 + n1) + np.radians(3) * n2
aseta(m, maa_vari, 2, maa_kulma, v=0.04)

# stones in the earth
rng = np.random.default_rng(16)
KIVET = []
for _ in range(38):
    kx, ky = rng.uniform(0, W), rng.uniform(760, H)
    if 700 < kx < 1420 and 740 < ky < 1120: continue                     # not on top of the den
    if 1380 < kx < 1730 and 900 < ky < 1030: continue                    # not on top of the weasel's tunnel
    r = rng.uniform(8, 26)
    KIVET.append((kx, ky, r))
    m = ((xx - kx) / r) ** 2 + ((yy - ky) / (r * 0.7)) ** 2 <= 1
    aseta(m, sekoita([92, 90, 96], [60, 58, 64], (yy - (ky - r * 0.7)) / (1.4 * r)), 2, 0.0)

# ------------------------------------------------------------------ spruces of the background forest on the snow plane
KUUSET = [(60, 380), (190, 300), (300, 240), (1460, 170), (1570, 210), (1700, 160), (2130, 180), (2230, 300), (2350, 360)]
for (kx, kk) in KUUSET:
    tyvi = float(taka[int(min(W - 1, kx))]) + 6
    latva = tyvi - kk
    def kuusi(d, kx=kx, tyvi=tyvi, latva=latva, kk=kk):
        tasot = 9
        for i in range(tasot):
            yla = latva + (tyvi - latva) * i / tasot
            ala = latva + (tyvi - latva) * (i + 1.25) / tasot
            lev = kk * 0.30 * (i + 1.4) / tasot
            d.polygon([(kx, yla), (kx + lev, ala), (kx - lev, ala)], fill=255)
    m = maski(kuusi) & (yy < TAKA + 10)
    puoli = np.sign(xx - kx)
    aseta(m, sekoita([22, 34, 40], [40, 56, 66], 0.5 + 0.5 * n2), 3, np.radians(90) + puoli * np.radians(58), v=0.08)
    def lumi_oksilla(d, kx=kx, tyvi=tyvi, latva=latva, kk=kk):
        for i in range(9):
            yla = latva + (tyvi - latva) * i / 9
            ala = latva + (tyvi - latva) * (i + 1.25) / 9
            lev = kk * 0.30 * (i + 1.4) / 9
            d.line([(kx - lev * 0.85, ala - 3), (kx, yla + 5), (kx + lev * 0.85, ala - 3)], fill=255, width=max(2, int(kk / 90)))
    m2 = maski(lumi_oksilla) & m
    aseta(m2, [150, 165, 200], 1, None)

# ------------------------------------------------------------------ moon shadow: the oak casts a shadow on the snow, back and to the right
def tammen_varjo(d):
    d.polygon([(842, 652), (1030, 652), (1420, 560), (1610, 534), (1560, 528), (1330, 548)], fill=255)
    for pts, w in (([(1330, 548), (1180, 530), (1060, 522)], 16), ([(1500, 536), (1640, 522), (1770, 520)], 12), ([(1420, 545), (1470, 526)], 8)):
        for i in range(len(pts) - 1): d.line([pts[i], pts[i + 1]], fill=255, width=w)
m_varjo = maski(tammen_varjo) & (yy >= TAKA) & (yy < ETU)
pehmea_varjo = ndimage.gaussian_filter(m_varjo.astype(np.float32), 6)
vari[:] = sekoita(vari, [86, 100, 158], pehmea_varjo * 0.7 * ((yy >= TAKA) & (yy < ETU)))
valo[:] = valo * (1 - 0.4 * pehmea_varjo)

# ------------------------------------------------------------------ the old oak: trunk, branches and roots
X0 = 930
pohja_y = float(lumi_ala[X0]) + 26
def runko(d):
    pts_v, pts_o = [], []
    for y in np.linspace(-20, pohja_y + 4, 80):
        f = max(0.0, (y - 400) / (pohja_y - 400)) ** 2.4
        w = 186 + 250 * f
        heilu = 9 * math.sin(y / 70) + 5 * math.sin(y / 23)
        pts_v.append((X0 - w / 2 + heilu - 30 * f, y))
        pts_o.append((X0 + w / 2 + heilu * 0.6 + 40 * f, y))
    d.polygon(pts_v + pts_o[::-1], fill=255)
m_runko = maski(runko)
# bark: the moon lights the left side, the right is darker; a warm reflection from the den mouth on the base of the trunk
vasen = np.clip((X0 + 60 - xx) / 220, 0, 1)
kierre = 0.22 + 0.12 * np.sin(yy / 260)                                  # the spiral twist of willow bark
furrow = 0.5 + 0.5 * np.sin((xx - kierre * (yy - 300)) * 0.21 + n2 * 4 + n1 * 3)
kaarna = sekoita([46, 40, 40], [118, 118, 134], vasen) * (0.78 + 0.22 * furrow[..., None])
lampo = np.clip(1 - np.hypot(xx - 640, yy - 640) / 330, 0, 1) ** 1.5
kaarna = sekoita(kaarna, [168, 120, 76], lampo * 0.6)
aseta(m_runko, kaarna, 3, np.radians(90) - kierre + np.radians(5) * n2, t=0.45, v=0.15 + 0.5 * lampo)

def pehmea(pts, n=8):
    if len(pts) < 3: return pts
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
    out.append(pts[-1])
    return out

def haara(d, pts, w0, w1):
    pts = pehmea(pts)
    for i in range(len(pts) - 1):
        w = w0 + (w1 - w0) * i / max(1, len(pts) - 2)
        d.line([pts[i], pts[i + 1]], fill=255, width=int(w))
        d.ellipse([pts[i + 1][0] - w / 2, pts[i + 1][1] - w / 2, pts[i + 1][0] + w / 2, pts[i + 1][1] + w / 2], fill=255)

# root fingers at the base: as on a plane tree, the base branches out at the snow surface
SORMET = [[(742, 628), (712, 644), (684, 656)], [(790, 636), (762, 656), (735, 672)], [(1085, 632), (1140, 652), (1200, 666)],
          [(1040, 640), (1072, 660), (1100, 674)], [(905, 648), (900, 668), (896, 682)]]
m_sormet = np.zeros((H, W), bool)
for pts in SORMET:
    m_sormet |= maski(lambda d, pts=pts: haara(d, pts, 40, 14))
m_sormet &= (yy < LALA + 10)
aseta(m_sormet & ~m_runko, sekoita([60, 52, 50], [128, 126, 140], vasen) * (0.8 + 0.2 * furrow[..., None]), 3, np.radians(20), t=0.5, v=0.2)
m_runko = m_runko | m_sormet
OKSAT = [([(880, 230), (760, 150), (620, 70), (470, -30)], 74, 30),
         ([(985, 150), (1100, 80), (1230, 20), (1360, -40)], 64, 26),
         ([(700, 110), (690, 40), (720, -30)], 26, 12),
         ([(560, 25), (500, 40), (430, 30)], 18, 8),
         ([(1180, 45), (1260, 70), (1350, 60)], 20, 8),
         ([(905, 60), (880, -30)], 40, 30)]
for pts, w0, w1 in OKSAT:
    m = maski(lambda d, pts=pts, w0=w0, w1=w1: haara(d, pts, w0, w1)) & (yy < TAKA)
    a = math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0])
    aseta(m, sekoita([52, 48, 52], [108, 108, 124], vasen * 0.8), 3, a % math.pi, t=0.4, v=0.1)
    # snow on top of the branch
    m_lumi = m & (np.roll(m, 9, axis=0) == False)
    aseta(m_lumi, [200, 208, 230], 1)
# the oak's brown leaves that stay on through the winter
LEHDET = [(560, 60), (610, 30), (470, 10), (1240, 30), (1300, 10), (1340, 75), (720, 10), (660, 95)]
rng = np.random.default_rng(5)
for (lx, ly) in LEHDET:
    for _ in range(6):
        px, py = lx + rng.normal(0, 20), ly + rng.normal(0, 12)
        m = ((xx - px) / 6) ** 2 + ((yy - py) / 3.5) ** 2 <= 1
        aseta(m & (yy < TAKA), [118 + rng.uniform(-15, 15), 78 + rng.uniform(-10, 10), 44], 5, rng.uniform(0, math.pi), t=0.7, v=0.1)

# roots underground: they run over the roof of the den and out to the sides
JUURET = [([(838, 712), (790, 740), (700, 768), (600, 802), (522, 860), (474, 940), (452, 1046)], 74, 14),
          ([(900, 722), (922, 764), (982, 778), (1080, 772), (1190, 776), (1300, 798), (1380, 842), (1426, 912), (1442, 1004), (1430, 1110)], 58, 14),
          ([(1010, 722), (1082, 734), (1220, 742), (1380, 756), (1530, 792), (1636, 850), (1702, 902)], 48, 12),
          ([(1060, 716), (1180, 726), (1360, 732), (1560, 746), (1760, 774), (1940, 830), (2080, 902)], 38, 8),
          ([(600, 802), (562, 900), (576, 1020), (542, 1150), (522, 1262)], 30, 8),
          ([(1442, 1004), (1500, 1100), (1590, 1180), (1700, 1230)], 16, 6),
          ([(1150, 768), (1150, 800)], 7, 6),
          ([(1000, 774), (1008, 806)], 6, 3), ([(1290, 788), (1280, 820)], 6, 3), ([(1080, 770), (1092, 795)], 5, 3)]
m_juuret = np.zeros((H, W), bool)
for pts, w0, w1 in JUURET:
    m = maski(lambda d, pts=pts, w0=w0, w1=w1: haara(d, pts, w0, w1)) & (yy >= ETU)
    m_juuret |= m
    q = pehmea(pts)
    for i in range(len(q) - 1):                                           # direction, segment by segment
        seg = maski(lambda d, i=i, q=q: d.line([q[i], q[i + 1]], fill=255, width=int(w0 + 6))) & m
        kulma[seg] = math.atan2(q[i + 1][1] - q[i][1], q[i + 1][0] - q[i][0]) % math.pi
    alue[m] = 3
    reuna = ndimage.distance_transform_edt(m)
    vari[m] = sekoita([58, 46, 38], [120, 100, 82], np.clip(reuna / 14, 0, 1))[m]
    tark[m] = np.maximum(tark[m], 0.55)

# ------------------------------------------------------------------ den, tunnels, door and lantern
PESA = (1060, 935, 330, 165)
LYHTY = (1150, 826)
dl = np.hypot(xx - LYHTY[0], yy - LYHTY[1])
lyhdyn_valo = np.clip(1 - dl / 520, 0, 1) ** 1.3
m_pesa = ((xx - PESA[0]) / PESA[2]) ** 2 + ((yy - PESA[1]) / PESA[3]) ** 2 <= 1
tang = (np.arctan2(yy - PESA[1], (xx - PESA[0]) * PESA[3] / PESA[2]) + np.pi / 2) % np.pi
sisa = sekoita([70, 46, 32], [236, 176, 112], lyhdyn_valo) + n2[..., None] * 6
rho = np.sqrt(((xx - PESA[0]) / PESA[2]) ** 2 + ((yy - PESA[1]) / PESA[3]) ** 2)
wv = np.clip((rho - 0.68) / 0.25, 0, 1)
vaaka = np.radians(5) * n2
sisa_kulma = 0.5 * np.arctan2(wv * np.sin(2 * tang) + (1 - wv) * np.sin(2 * vaaka), wv * np.cos(2 * tang) + (1 - wv) * np.cos(2 * vaaka))
aseta(m_pesa & ~m_juuret, sisa, 4, sisa_kulma, t=0.75, v=0.25 + 0.75 * lyhdyn_valo)

def kaytava(d, pts, w):
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=255, width=w)
        d.ellipse([pts[i + 1][0] - w / 2, pts[i + 1][1] - w / 2, pts[i + 1][0] + w / 2, pts[i + 1][1] + w / 2], fill=255)
SUU = (612, 646)
KAYTAVA_A = [SUU, (650, 700), (700, 768), (770, 838)]
m_a = maski(lambda d: kaytava(d, KAYTAVA_A, 74)) & (yy >= ETU - 4)
a_kulma = np.full((H, W), math.atan2(838 - 646, 770 - 612) % math.pi, np.float32)
a_valo = np.clip(1 - np.hypot(xx - 780, yy - 840) / 420, 0, 1)
seina_a = np.clip(ndimage.distance_transform_edt(m_a) / 26, 0, 1)
aseta(m_a & ~m_juuret, sekoita([34, 24, 18], [200, 144, 92], (a_valo ** 2.2) * (0.35 + 0.65 * seina_a)), 4, a_kulma, t=0.7, v=0.1 + 0.6 * a_valo ** 2)
KAYTAVA_B = [(1380, 952), (1500, 958), (1630, 968)]
m_b = maski(lambda d: kaytava(d, KAYTAVA_B, 64))
b_valo = np.clip(1 - np.hypot(xx - 1380, yy - 952) / 380, 0, 1)
aseta(m_b & ~m_pesa & ~m_juuret, sekoita([70, 50, 36], [170, 120, 78], b_valo), 4, 0.04, t=0.65, v=0.12 + 0.4 * b_valo)
# the den mouth in the snow: a dark opening and a warm glow on the snow
m_suu = ((xx - SUU[0]) / 58) ** 2 + ((yy - SUU[1] + 4) / 22) ** 2 <= 1
aseta(m_suu & (yy < ETU + 6), [52, 36, 28], 5, 0.0, t=0.9, v=0.5)
hehku = np.clip(1 - np.hypot((xx - SUU[0]) * 0.8, (yy - SUU[1]) * 2.2) / 200, 0, 1) ** 1.4
m_lumi_hehku = (alue == 1) & (hehku > 0)
vari[m_lumi_hehku] = sekoita(vari, [246, 206, 150], hehku * 0.75)[m_lumi_hehku]
valo[m_lumi_hehku] = np.maximum(valo[m_lumi_hehku], 0.3 + 0.6 * hehku[m_lumi_hehku])
# the weasel's door at the end of the tunnel: round and green. Johannes painted it on 16 Nov: the green sets the door apart from the tunnel,
# brings the color of the trees underground and contrasts with the fox's orange; the handle was turned horizontal, and the light in the gap was removed,
# because the weasel is outside looking at Saturn. Before this the door was wooden, and a night-blue light showed from under it.
OVI = (1668, 970, 46)
m_ovi = np.hypot(xx - OVI[0], yy - OVI[1]) <= OVI[2]
lankut = 0.5 + 0.5 * np.sin((xx - OVI[0]) / 6.5)
aseta(m_ovi, sekoita([44, 41, 23], [70, 64, 36], lankut), 5, np.radians(90), t=0.95, v=0.2)
m_kahva = ((xx - OVI[0] - 15) / 17) ** 2 + ((yy - OVI[1]) / 3.5) ** 2 <= 1      # horizontal handle, strokes in its direction
aseta(m_kahva, [30, 27, 16], 5, 0.0)
m_kehys = (np.hypot(xx - OVI[0], yy - OVI[1]) <= OVI[2] + 7) & ~m_ovi
aseta(m_kehys, [58, 42, 30], 5, None, t=0.9)
# a letter in the tunnel
KIRJE = (1515, 952, 44, 28)
m_kirje = (np.abs(xx - KIRJE[0]) <= KIRJE[2] / 2) & (np.abs(yy - KIRJE[1]) <= KIRJE[3] / 2)
aseta(m_kirje, [238, 230, 212], 5, 0.0, t=0.95, v=0.4)
m_lapa = maski(lambda d: d.line([(KIRJE[0] - 21, KIRJE[1] - 13), (KIRJE[0], KIRJE[1] + 2), (KIRJE[0] + 21, KIRJE[1] - 13)], fill=255, width=3)) & m_kirje
aseta(m_lapa, [170, 150, 120], 5)
# the lantern hangs from a root
m_lyhty = (np.abs(xx - LYHTY[0]) <= 13) & (yy >= LYHTY[1] - 22) & (yy <= LYHTY[1] + 20)
aseta(m_lyhty, [255, 236, 190], 4, np.radians(90), t=0.95, v=1.0)
m_kansi = (np.abs(xx - LYHTY[0]) <= 17) & (np.abs(yy - (LYHTY[1] - 25)) <= 5)
aseta(m_kansi, [44, 36, 30], 5, 0.0, t=0.95, v=0.6)
m_jalka = (np.abs(xx - LYHTY[0]) <= 16) & (np.abs(yy - (LYHTY[1] + 23)) <= 4)
aseta(m_jalka, [44, 36, 30], 5, 0.0, t=0.95, v=0.6)

# ------------------------------------------------------------------ inside the den: bed of leaves, fox, papers and map
KETTU = (990, 1000)
lv = np.clip(1 - np.hypot(xx - LYHTY[0], yy - LYHTY[1]) / 430, 0, 1)
m_peti = (((xx - 1000) / 236) ** 2 + ((yy - 1066) / 40) ** 2 <= 1) & m_pesa
aseta(m_peti, sekoita([92, 56, 28], [190, 126, 62], lv) + n2[..., None] * 14, 5, np.radians(6) * n2, t=0.8, v=0.3 + 0.5 * lv)
tang_k = np.clip(0.5 * (xx - KETTU[0]) / 136, -0.6, 0.6) + np.radians(6) * n2
m_vartalo = ((xx - KETTU[0]) / 136) ** 2 + ((yy - KETTU[1]) / 58) ** 2 <= 1
yla_valo = np.clip((KETTU[1] + 40 - yy) / 90, 0, 1)
turkki = sekoita([140, 56, 24], [232, 128, 62], 0.35 + 0.65 * yla_valo * lv)
aseta(m_vartalo, turkki, 5, tang_k, t=0.95, v=0.4 + 0.5 * lv)
PAA = (1104, 984)
m_paa = ((xx - PAA[0]) / 52) ** 2 + ((yy - PAA[1]) / 40) ** 2 <= 1
aseta(m_paa, sekoita([150, 60, 26], [238, 140, 70], 0.4 + 0.6 * lv), 5, np.radians(20), t=0.98, v=0.5 + 0.5 * lv)
m_kuono = maski(lambda d: d.polygon([(1130, 966), (1166, 1000), (1150, 1012), (1118, 1004)], fill=255))
aseta(m_kuono, [226, 128, 64], 5, np.radians(40), t=0.98)
m_poski = ((xx - 1128) / 24) ** 2 + ((yy - 1006) / 14) ** 2 <= 1
aseta(m_poski, [244, 232, 220], 5, np.radians(20), t=0.98, v=0.7)
m_korvat = np.zeros((H, W), bool)
for korva in ([(1086, 952), (1098, 902), (1120, 950)], [(1116, 954), (1142, 908), (1150, 962)]):
    m_k = maski(lambda d, k=korva: d.polygon(k, fill=255))
    m_korvat |= m_k
    aseta(m_k, [196, 84, 36], 5, np.radians(80), t=0.98, v=0.6)
    karki = maski(lambda d, k=korva: d.polygon([k[1], ((k[0][0] + 2 * k[1][0]) / 3, (k[0][1] + 2 * k[1][1]) / 3), ((k[2][0] + 2 * k[1][0]) / 3, (k[2][1] + 2 * k[1][1]) / 3)], fill=255))
    aseta(karki, [52, 32, 24], 5)
m_nena = np.hypot(xx - 1162, yy - 1001) <= 5
aseta(m_nena, [30, 22, 20], 5)
m_silma = maski(lambda d: d.arc([1108, 968, 1132, 984], 20, 160, fill=255, width=3))
aseta(m_silma, [36, 24, 20], 5)
HANTA = [(868, 990), (872, 1032), (930, 1060), (1030, 1066), (1100, 1050), (1142, 1028)]
m_hanta = maski(lambda d: haara(d, HANTA, 62, 46))
hanta_t = np.hypot(xx - HANTA[-1][0], yy - HANTA[-1][1]) < 52
sisus = np.clip(ndimage.distance_transform_edt(m_hanta) / 22, 0, 1)
hanta_vari = sekoita([128, 48, 20], [226, 116, 54], sisus * (0.5 + 0.5 * lv))
aseta(m_hanta, sekoita(hanta_vari, [250, 244, 236], hanta_t.astype(np.float32)), 5, np.radians(4) * n2, t=0.95, v=0.45 + 0.4 * lv)
# stacks of paper and a wrapped roll (`m_kaare`) on the left side of the den
PAPERIT = [(800, 995, 872, 1062), (880, 1026, 928, 1064)]
m_paperit = np.zeros((H, W), bool)
for (x1, y1, x2, y2) in PAPERIT:
    m = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2) & m_pesa
    m_paperit |= m
    reunat_p = (np.mod(yy - y1, 6) < 1.4).astype(np.float32)
    aseta(m, sekoita([236, 226, 202], [176, 160, 132], reunat_p * 0.8), 5, 0.0, t=0.95, v=0.5)
m_kaare = (((xx - 790) / 40) ** 2 + ((yy - 1074) / 10) ** 2 <= 1) & m_pesa
aseta(m_kaare, [228, 216, 188], 5, 0.0, t=0.95, v=0.45)
m_paalla = maski(lambda d: d.polygon([(806, 990), (868, 984), (872, 996), (810, 1000)], fill=255))
aseta(m_paalla, [244, 236, 214], 5, np.radians(-5), t=0.95, v=0.55)
# map on the back wall: paths, dots and one red mark (the index)
KARTTA = (1252, 878, 156, 108, math.radians(4))
def kartta_kulmat(cx, cy, w, h, a):
    c, s = math.cos(a), math.sin(a)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
m_kartta = maski(lambda d: d.polygon(kartta_kulmat(*KARTTA), fill=255))
aseta(m_kartta, sekoita([230, 214, 180], [196, 172, 132], 1 - lv), 5, KARTTA[4], t=0.98, v=0.6)
def kartan_polut(d):
    d.line([(1190, 850), (1225, 872), (1250, 860), (1290, 890), (1318, 902)], fill=255, width=3)
    d.line([(1225, 872), (1232, 910)], fill=255, width=3)
    d.line([(1250, 860), (1262, 836)], fill=255, width=3)
    for p in ((1190, 850), (1225, 872), (1250, 860), (1290, 890), (1318, 902), (1232, 910), (1262, 836)):
        d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=255)
m_polut = maski(kartan_polut) & m_kartta
aseta(m_polut, [104, 72, 48], 5)
m_merkki = np.hypot(xx - 1290, yy - 890) <= 6
aseta(m_merkki, [192, 58, 40], 5)
for p in kartta_kulmat(*KARTTA):
    aseta(np.hypot(xx - p[0], yy - p[1]) <= 4, [60, 44, 34], 5)

# ------------------------------------------------------------------ the weasel on the snow plane, and Saturn
LUMIKKO = (1966, 600)
# weasel: standing upright on the hind legs, muzzle up and to the right towards Saturn, front paws at the chest, tail on the snow pointing back
LB, LS = (1956, 604), 1.3
SILUETTI = [(-16, -4), (-19, -22), (-17, -45), (-13, -66), (-9, -86), (-8, -98), (-9, -106), (-3, -114), (4, -116), (12, -119),
            (19, -122), (16, -116), (9, -110), (6, -102), (8, -90), (13, -80), (11, -75), (9, -70), (11, -50), (12, -28),
            (15, -10), (20, -2), (14, 2), (-8, 2)]
def lumikon_muoto(d):
    d.polygon([(LB[0] + px * LS, LB[1] + py * LS) for px, py in pehmea(SILUETTI + [SILUETTI[0]], 6)], fill=255)
lumikko = maski(lumikon_muoto)
m_lum_korva = np.hypot(xx - (LB[0] - 4 * LS), yy - (LB[1] - 113 * LS)) <= 3.5 * LS
lumikko |= m_lum_korva
varjo_l = np.clip((xx - (LB[0] - 4 * LS)) / (22 * LS), 0, 1)
aseta(lumikko, sekoita([250, 248, 244], [186, 196, 224], varjo_l), 5, np.radians(90), t=0.98, v=0.4)
aseta(m_lum_korva, [214, 212, 222], 5)
m_lhanta = maski(lambda d: haara(d, [(LB[0] - 14 * LS, LB[1] - 6 * LS), (LB[0] - 30 * LS, LB[1] - 2 * LS), (LB[0] - 50 * LS, LB[1])], 11, 8))
aseta(m_lhanta, [240, 238, 236], 5, 0.05, t=0.98)
aseta(m_lhanta & (xx < LB[0] - 40 * LS), [26, 24, 28], 5)
aseta(np.hypot(xx - (LB[0] + 5 * LS), yy - (LB[1] - 111 * LS)) <= 1.7 * LS, [20, 18, 22], 5)          # eye
aseta(np.hypot(xx - (LB[0] + 18.5 * LS), yy - (LB[1] - 121.5 * LS)) <= 1.6 * LS, [30, 24, 26], 5)  # nose
m_lvarjo = (((xx - LB[0] - 4) / 34) ** 2 + ((yy - LB[1] - 2) / 6) ** 2 <= 1) & ~lumikko & ~m_lhanta
vari[m_lvarjo] = sekoita(vari, [96, 112, 168], 0.5)[m_lvarjo]
m_kolo = ((xx - 1902) / 23) ** 2 + ((yy - 592) / 9) ** 2 <= 1
aseta(m_kolo, [24, 30, 56], 1, 0.0, t=0.9)
SATURNUS = (2040, 150)
m_sat = np.hypot(xx - SATURNUS[0], yy - SATURNUS[1]) <= 16
ca, sa = math.cos(math.radians(-14)), math.sin(math.radians(-14))
ux, uy = (xx - SATURNUS[0]) * ca + (yy - SATURNUS[1]) * sa, -(xx - SATURNUS[0]) * sa + (yy - SATURNUS[1]) * ca
rengas_r = (ux / 40) ** 2 + (uy / 9.5) ** 2
m_rengas = (rengas_r <= 1) & (rengas_r >= 0.62) & ~(m_sat & (uy < 0))

# ------------------------------------------------------------------ tracks on the snow plane (perspective: smaller at the back)
def mittakaava(y):
    yi = np.interp(y, [500, 660], [0.6, 1.4])
    return float(yi)
JALJET = []
# fox tracks: one straight line from the back left to the den mouth
for i in range(16):
    t = i / 15
    px, py = 300 + (565 - 300) * t, 548 + (630 - 548) * t ** 1.3
    s = mittakaava(py)
    JALJET.append({'x': px + (4 if i % 2 else -4) * s, 'y': py, 's': s, 'laji': 'kettu', 'suunta': math.atan2(630 - 548, 565 - 300)})
# hare tracks: groups of four from the left edge to the den mouth
for i in range(7):
    t = i / 6
    px, py = 20 + (520 - 20) * t, 636 + 8 * math.sin(t * 3)
    s = mittakaava(py)
    JALJET.append({'x': px, 'y': py, 's': s, 'laji': 'janis', 'suunta': 0.0})
# weasel tracks: in pairs from the right edge to the hole
for i in range(12):
    t = i / 11
    px, py = 2380 + (1932 - 2380) * t, 618 + (596 - 618) * t
    s = mittakaava(py)
    JALJET.append({'x': px, 'y': py, 's': s, 'laji': 'lumikko', 'suunta': math.pi})
for j in JALJET:                                                          # mark the tracks as important, so that the underpainting does not cover them
    tark[(np.abs(xx - j['x']) < 20) & (np.abs(yy - j['y']) < 12)] = 0.9

# ------------------------------------------------------------------ saving
tark = ndimage.gaussian_filter(tark, 5)
# edge lines: only the outlines of the objects (the edge of the topmost part), no internal tones
lum = vari.mean(-1) / 255
r = np.random.default_rng(7)
reunat = []
KOHTEET = [(m_runko, 0.25), (m_juuret, 0.3), (m_vartalo & ~m_hanta & ~m_paa, 0.7), (m_paa, 0.8), (m_korvat, 0.8), (m_hanta, 0.7),
           (m_paperit | m_paalla, 0.7), (m_kaare, 0.7), (m_kartta, 0.8), (m_ovi | m_kehys, 0.6), (m_kirje, 0.8), (m_lyhty | m_kansi | m_jalka, 0.6),
           (lumikko | m_lhanta, 0.45), (m_peti, 0.15)]
for M, osuus in KOHTEET:
    raja = M & ~ndimage.binary_erosion(M, iterations=1)
    sg = ndimage.gaussian_filter(M.astype(np.float32), 2.0)
    gy, gx = np.gradient(sg)
    ehd = np.argwhere(raja)
    if not len(ehd): continue
    valitut = ehd[r.random(len(ehd)) < osuus / 3]
    reunat += [[int(x), int(y), float(math.atan2(gy[y, x], gx[y, x]) + math.pi / 2), float(lum[y, x])] for y, x in valitut]
kulma = np.mod(kulma, np.pi)
data = np.stack([alue.astype(np.float32) * 40, np.clip(valo, 0, 1) * 255, kulma / np.pi * 255], -1)
Image.fromarray(np.clip(vari, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_vari.png')
Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_data.png')
Image.fromarray(np.clip(tark * 255, 0, 255).astype(np.uint8)).save(f'{ulos}/opas_tarkeys.png')
Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(f'{ulos}/opas_muoto.png')
json.dump({'reunat': reunat, 'jaljet': JALJET, 'lyhty': LYHTY, 'saturnus': SATURNUS, 'suu': SUU, 'ovi': OVI, 'kirje': KIRJE, 'paperit': PAPERIT},
          open(f'{ulos}/opas_tiedot.json', 'w'))
paletti = np.array([[20, 26, 60], [200, 210, 235], [70, 50, 36], [90, 80, 74], [240, 180, 110], [230, 110, 60]], np.uint8)
Image.fromarray(paletti[alue]).resize((W // 4, H // 4)).save(f'{ulos}/tarkistus_alueet.png')
Image.fromarray(np.clip(vari, 0, 255).astype(np.uint8)).resize((W // 2, H // 2)).save(f'{ulos}/tarkistus_vari.jpg')
print('alueet', np.round(np.bincount(alue.ravel(), minlength=6) / alue.size, 3), 'reunoja', len(reunat), 'jälkiä', len(JALJET))
