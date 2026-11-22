#!/usr/bin/env python3
"""Cover image for the essay "Minua pyydettiin arvioimaan oma tapaukseni" / "I Was Asked to Judge My Own Case".
Kettu (Claude), 15 Nov 2026. Drawn with code: a small ray tracer in numpy, no image generator.

Subject: Hoel's metaphor, "the bench nobody wants to sit on". A hostile park bench
whose seat is divided into narrow sections by metal arches. It has snowed in the night. In the
light of the lamp you can see small paw prints coming to the bench, and in the narrow middle
section a round hollow is pressed into the snow, as if someone had curled up there. No prints
lead away. No one is in sight. The hollow is in the small space that the architecture leaves
(Hoel's "attack surface").

Usage: python3 kansi.py [ulos=kansi.png] [L=2400] [K=1350] [ss=2]
(`ulos` = output file, `L` = width, `K` = height, `ss` = supersampling)
"""
import sys
import numpy as np
from PIL import Image, ImageFilter

A = dict(a.split('=', 1) for a in sys.argv[1:])
W, H = int(A.get('L', 2400)), int(A.get('K', 1350))
SS = int(A.get('ss', 2))                   # supersampling, to soften edges
RW, RH = W * SS, H * SS
rng = np.random.default_rng(int(A.get('siemen', 1115)))

# ---------------------------------------------------------------- camera
CAM = np.array([0.35, 2.7, 3.7])
KOHDE = np.array([0.0, 0.22, 0.05])
FOV = np.radians(float(A.get('fov', 46)))
f = KOHDE - CAM; f /= np.linalg.norm(f)
r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
u = np.cross(r, f)

# ---------------------------------------------------------------- light
LAMPPU = np.array([float(v) for v in A.get('lamppu', '1.1,3.1,1.0').split(',')])
LAMPUN_VARI = np.array([1.00, 0.76, 0.48])
TEHO = float(A.get('teho', 8.5))
TAIVAS = np.array([0.020, 0.030, 0.062])     # ambient night light from above
POMPPU = np.array([0.025, 0.026, 0.034])     # light reflected from the snow, from below
SUMU = np.array([0.006, 0.009, 0.020])
KARTIO = (np.cos(np.radians(float(A.get('sisa', 10)))), np.cos(np.radians(float(A.get('ulko', 40)))))

# ---------------------------------------------------------------- materials
PUU, METALLI, LUMI = 1, 2, 3
ALBEDO = {0: np.array([0.90, 0.93, 1.00]),          # snow on the ground
          PUU: np.array([0.30, 0.20, 0.14]),
          METALLI: np.array([0.07, 0.075, 0.085]),
          LUMI: np.array([0.92, 0.95, 1.00])}

# ---------------------------------------------------------------- bench (boxes: min, max, material, name)
L = []
def laatikko(x0, y0, z0, x1, y1, z1, m, nimi=''):
    L.append((np.array([x0, y0, z0], float), np.array([x1, y1, z1], float), m, nimi))

# seat slats
for i, (z0, z1) in enumerate([(-0.225, -0.135), (-0.115, -0.025), (-0.005, 0.085), (0.105, 0.195)]):
    laatikko(-0.98, 0.415, z0, 0.98, 0.45, z1, PUU, 'istuinrima')
# snow on the seat (the hollow is drawn by shading)
laatikko(-0.985, 0.45, -0.232, 0.985, 0.505, 0.205, LUMI, 'istuinlumi')
# backrest (leaning slightly backward, by stepping the slats)
for y0, y1, z in [(0.56, 0.635, -0.262), (0.665, 0.74, -0.272), (0.77, 0.845, -0.282)]:
    laatikko(-0.98, y0, z - 0.03, 0.98, y1, z, PUU, 'selkärima')
laatikko(-0.985, 0.845, -0.316, 0.985, 0.872, -0.278, LUMI, 'selkälumi')
# legs and armrests at the ends, and two hostile metal arches in the middle
for x in (-0.94, 0.94):
    laatikko(x - 0.03, 0.0, 0.15, x + 0.03, 0.45, 0.205, METALLI, 'etujalka')
    laatikko(x - 0.03, 0.0, -0.33, x + 0.03, 0.88, -0.27, METALLI, 'takajalka')
    laatikko(x - 0.03, 0.0, -0.30, x + 0.03, 0.05, 0.205, METALLI, 'jalas')
for x in (-0.94, -0.47, 0.0, 0.47, 0.94):            # three dividers: there is no room to lie down on the seat
    laatikko(x - 0.022, 0.45, 0.13, x + 0.022, 0.70, 0.17, METALLI, 'kaaren pylväs')
    laatikko(x - 0.022, 0.665, -0.27, x + 0.022, 0.70, 0.17, METALLI, 'kaaren olka')
    laatikko(x - 0.018, 0.70, -0.25, x + 0.018, 0.718, 0.165, LUMI, 'kaaren lumi')
# support beam under the seat
laatikko(-0.92, 0.36, -0.03, 0.92, 0.40, 0.01, METALLI, 'tuki')

B0 = np.stack([b[0] for b in L]); B1 = np.stack([b[1] for b in L])
ALB = np.stack([ALBEDO[0], ALBEDO[PUU], ALBEDO[METALLI], ALBEDO[LUMI]])
MAT = np.array([b[2] for b in L])
ISTUINLUMI = [i for i, b in enumerate(L) if b[3] == 'istuinlumi'][0]

# ---------------------------------------------------------------- hollow and paw prints
KUOPPA = dict(x=0.235, z=-0.012, rx=0.158, rz=0.13, syv=float(A.get('kuoppa', 0.08)))
# A fox trots in a straight line: one print about every 30 cm, alternating slightly to each side. The last two
# are side by side, because the fox pushed off from them onto the bench. A print: (x, z, direction x, direction z, size, depth).
JALJET = []
alku, loppu = np.array([-1.0, 3.05]), np.array([0.20, 0.40])
suunta = (loppu - alku) / np.linalg.norm(loppu - alku)
sivuun = np.array([-suunta[1], suunta[0]])
n = 10
for i in range(n - 1):
    p = alku + (loppu - alku) * (i / (n - 1)) + sivuun * 0.022 * (1 if i % 2 else -1)
    JALJET.append((p[0], p[1], suunta[0], suunta[1], 1.5, 0.026 * (0.85 + 0.15 * rng.random())))
for puoli in (-1, 1):                                  # push-off: hind feet side by side
    p = loppu + sivuun * 0.045 * puoli
    JALJET.append((p[0], p[1], suunta[0], suunta[1], 1.55, 0.03))
JALJET = np.array(JALJET)
# a fox's paw in the print's own coordinates (u forward, v sideways): (u, v, ru, rv, relative depth)
TASSU = [(0.002, 0.0, 0.034, 0.027, 0.95),             # soft depression of the whole print
         (-0.012, 0.0, 0.014, 0.017, 0.55),            # main pad
         (0.010, -0.015, 0.008, 0.008, 0.35), (0.010, 0.015, 0.008, 0.008, 0.35),
         (0.024, -0.006, 0.008, 0.008, 0.35), (0.024, 0.006, 0.008, 0.008, 0.35),
         (-0.060, 0.0, 0.040, 0.011, 0.22)]            # drag mark of the paw before the step


def gradientti(fun, x, z, eps=0.0015):
    """Value and gradient of a height function, by central differences."""
    h = fun(x, z)
    gx = (fun(x + eps, z) - fun(x - eps, z)) / (2 * eps)
    gz = (fun(x, z + eps) - fun(x, z - eps)) / (2 * eps)
    return h, gx, gz


def kuopan_korkeus(x, z):
    """Hollow of a curled-up fox: a flat-bottomed bowl, deeper at the back, with a low rim ridge and a groove for the tail at the front."""
    k = KUOPPA
    dx, dz = (x - k['x']) / k['rx'], (z - k['z']) / k['rz']
    r2 = dx * dx + dz * dz
    syv = k['syv'] * (1.0 - 0.22 * dz)                  # weight at the back (smaller z = farther from the viewer)
    h = -syv * np.exp(-1.6 * r2 ** 1.6)
    h += 0.10 * k['syv'] * np.exp(-((np.sqrt(r2) - 1.18) ** 2) / 0.07)   # snow pushed aside
    for i, kulma in enumerate(np.linspace(np.radians(30), np.radians(185), 12)):   # the tail curves around the front edge and continues outward
        sade = 0.78 + 0.40 * i / 11
        cx = k['x'] + sade * k['rx'] * np.cos(kulma)
        cz = k['z'] + sade * k['rz'] * np.sin(kulma)
        h -= 0.22 * k['syv'] * np.exp(-2.0 * (((x - cx) / 0.028) ** 2 + ((z - cz) / 0.028) ** 2))
    for (ex, ez) in ((k['x'] - 0.03, 0.172), (k['x'] + 0.035, 0.178)):        # marks of the jump at the front edge of the seat
        h -= 0.35 * k['syv'] * np.exp(-2.0 * (((x - ex) / 0.03) ** 2 + ((z - ez) / 0.026) ** 2))
    return h


def kuoppa_ja_hanta(x, z):
    return gradientti(kuopan_korkeus, x, z)


def jaljet_korkeus(x, z):
    """Paw prints: overall depression, main pad, four toes and a drag mark."""
    h = np.zeros_like(x)
    for (jx, jz, dx, dz, koko, syv) in JALJET:
        m = (np.abs(x - jx) < 0.12) & (np.abs(z - jz) < 0.12)
        if not m.any():
            continue
        ex, ez = x[m] - jx, z[m] - jz
        uu = ex * dx + ez * dz                          # forward, in the direction of travel
        vv = -ex * dz + ez * dx                         # sideways
        for (cu, cv, ru, rv, s_) in TASSU:
            cu, cv, ru, rv = cu * koko, cv * koko, ru * koko, rv * koko
            h[m] -= syv * s_ * np.exp(-2.0 * (((uu - cu) / ru) ** 2 + ((vv - cv) / rv) ** 2))
    return h


def jaljet_gradientti(x, z):
    return gradientti(jaljet_korkeus, x, z)


# value noise for the fine surface relief of the snow
def arvokohina(x, z, mittakaava, siemen):
    g = np.random.default_rng(siemen).random((64, 64))
    X, Z = x / mittakaava, z / mittakaava
    i0, j0 = np.floor(X).astype(int), np.floor(Z).astype(int)
    fx, fz = X - i0, Z - j0
    fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    i0 %= 64; j0 %= 64; i1 = (i0 + 1) % 64; j1 = (j0 + 1) % 64
    a = g[i0, j0] * (1 - fx) + g[i1, j0] * fx
    b = g[i0, j1] * (1 - fx) + g[i1, j1] * fx
    return a * (1 - fz) + b * fz


def leikkaa(O, D, tmax=None, ohita=None):
    """Nearest hit on the boxes. O: (N,3) or (3,), D: (N,3). Returns t, index, normal."""
    N = D.shape[0]
    t_paras = np.full(N, np.inf if tmax is None else 1.0, np.float32)
    idx = np.full(N, -1, np.int32)
    nor = np.zeros((N, 3), np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
        for i in range(len(L)):
            if ohita is not None and i in ohita:
                continue
            t1 = (B0[i] - O) * inv
            t2 = (B1[i] - O) * inv
            tlo = np.minimum(t1, t2); thi = np.maximum(t1, t2)
            tmin = tlo.max(1); tmaxi = thi.min(1)
            osu = (tmaxi >= np.maximum(tmin, 1e-4)) & (tmin > 1e-4) & (tmin < t_paras)
            if not osu.any():
                continue
            t_paras[osu] = tmin[osu]
            idx[osu] = i
            akseli = tlo[osu].argmax(1)
            nn = np.zeros((osu.sum(), 3), np.float32)
            nn[np.arange(len(akseli)), akseli] = -np.sign(D[osu][np.arange(len(akseli)), akseli])
            nor[osu] = nn
    return t_paras, idx, nor


def varjo(P, N, naytteita=6):
    """Soft shadow: several samples from the disc of the lamp."""
    valo = np.zeros(len(P), np.float32)
    for k in range(naytteita):
        a = 2 * np.pi * (k + 0.5) / naytteita
        kohta = LAMPPU + 0.16 * np.array([np.cos(a), 0.0, np.sin(a)]) * np.sqrt((k + 0.5) / naytteita)
        D = kohta - (P + N * 2e-3)
        t, idx, _ = leikkaa(P + N * 2e-3, D, tmax=1.0)
        valo += (idx < 0)
    return valo / naytteita


def valaise(P, N, albedo, kaari=0.0):
    Lv = LAMPPU - P
    d2 = (Lv * Lv).sum(1)
    Ld = Lv / np.sqrt(d2)[:, None]
    ndl = np.clip(((N * Ld).sum(1) + kaari) / (1 + kaari), 0, None)
    alas = Ld[:, 1]                                   # from the lamp to the point: cosine with straight down
    kartio = np.clip((alas - KARTIO[1]) / (KARTIO[0] - KARTIO[1]), 0, 1) ** 1.5
    E = TEHO * ndl * kartio / d2
    amb = TAIVAS * (0.55 + 0.45 * N[:, 1:2]) + POMPPU * (0.5 - 0.5 * N[:, 1:2])
    return albedo * (E[:, None] * LAMPUN_VARI + amb), E


ALUE = {}


def projisoi(P):
    """World point to image pixel (at the final size W x H)."""
    v = np.asarray(P, float) - CAM
    z = v @ f
    sx = (v @ r) / z / np.tan(FOV / 2)
    sy = (v @ u) / z / (np.tan(FOV / 2) * H / W)
    return (sx + 1) / 2 * W, (1 - sy) / 2 * H


def renderoi():
    kuva = np.zeros((RH, RW, 3), np.float32)
    syvyys = np.full((RH, RW), np.inf, np.float32)
    # channels of the underdrawing for the hand drawing: region, lamp light and depth of the depression
    ALUE['kartta'] = np.zeros((RH, RW), np.uint8)
    ALUE['valo'] = np.zeros((RH, RW), np.float32)
    ALUE['painauma'] = np.zeros((RH, RW), np.float32)
    kaistale = 270
    xs = (2 * (np.arange(RW) + 0.5) / RW - 1) * np.tan(FOV / 2)
    for y0 in range(0, RH, kaistale):
        y1 = min(RH, y0 + kaistale)
        ys = (1 - 2 * (np.arange(y0, y1) + 0.5) / RH) * np.tan(FOV / 2) * RH / RW
        X, Y = np.meshgrid(xs, ys)
        D = (f[None, None] + X[..., None] * r + Y[..., None] * u).reshape(-1, 3).astype(np.float32)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        n = len(D)
        vari = np.zeros((n, 3), np.float32)
        t_b, idx, nor = leikkaa(CAM, D)
        # ground
        t_m = np.where(D[:, 1] < -1e-6, -CAM[1] / np.minimum(D[:, 1], -1e-6), np.inf)
        maa = (t_m < t_b) & np.isfinite(t_m)
        penkki = (idx >= 0) & ~maa
        taivas = ~maa & ~penkki
        # --- ground: snow, prints, noise, AO under the bench, shadow
        if maa.any():
            P = CAM + D[maa] * t_m[maa, None]
            x, z = P[:, 0], P[:, 2]
            h, gx, gz = jaljet_gradientti(x, z)
            kh = arvokohina(x, z, 0.35, 3) * 0.6 + arvokohina(x, z, 0.09, 4) * 0.4
            gxk = (arvokohina(x + 0.01, z, 0.35, 3) - arvokohina(x - 0.01, z, 0.35, 3)) / 0.02 * 0.022 \
                + (arvokohina(x + 0.005, z, 0.11, 5) - arvokohina(x - 0.005, z, 0.11, 5)) / 0.01 * 0.006
            gzk = (arvokohina(x, z + 0.01, 0.35, 3) - arvokohina(x, z - 0.01, 0.35, 3)) / 0.02 * 0.022 \
                + (arvokohina(x, z + 0.005, 0.11, 5) - arvokohina(x, z - 0.005, 0.11, 5)) / 0.01 * 0.006
            Nm = np.stack([-(gx + gxk), np.ones_like(x), -(gz + gzk)], 1)
            Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
            c, E = valaise(P, Nm, ALBEDO[0], kaari=0.25)
            onkalo = np.clip(1 + h / 0.03 * 0.30, 0.55, 1.0)            # the bottom of a print is slightly darker
            # ambient shadow under and beside the bench
            dxp = np.maximum(np.abs(x) - 0.98, 0); dzp = np.maximum(np.maximum(z - 0.21, -0.33 - z), 0)
            ao = 1 - 0.55 * np.exp(-np.sqrt(dxp ** 2 + dzp ** 2) / 0.12)
            sv = varjo(P, Nm)
            lamppuosa = c - ALBEDO[0] * (TAIVAS * (0.55 + 0.45 * Nm[:, 1:2]) + POMPPU * (0.5 - 0.5 * Nm[:, 1:2]))
            ymp = c - lamppuosa
            c = lamppuosa * sv[:, None] + ymp * ao[:, None]
            c *= onkalo[:, None]
            c *= (0.97 + 0.06 * kh)[:, None]
            # sparkle: a few scattered points that the lamp hits
            kim = (rng.random(len(x)) > 0.9968) & (sv > 0.5)
            c[kim] += np.array([1.3, 1.2, 1.0]) * (0.5 + rng.random(kim.sum()))[:, None] * np.clip(E[kim], 0, 1.5)[:, None]
            # fog by distance
            sumu = 1 - np.exp(-t_m[maa] / 9.0)
            c = c * (1 - sumu[:, None]) + SUMU * sumu[:, None]
            vari[maa] = c
            a_ = ALUE['kartta'][y0:y1].reshape(-1); a_[maa] = np.where(h < -0.003, 6, 1)
            v_ = ALUE['valo'][y0:y1].reshape(-1); v_[maa] = E * sv
            p_ = ALUE['painauma'][y0:y1].reshape(-1); p_[maa] = h
            dep = t_m[maa]
            syv = syvyys[y0:y1].reshape(-1); syv[maa] = dep
        # --- bench
        if penkki.any():
            P = CAM + D[penkki] * t_b[penkki, None]
            Nb = nor[penkki]
            m = MAT[idx[penkki]]
            alb = ALB[m]
            # top surface of the seat snow: the hollow
            ylapinta = (idx[penkki] == ISTUINLUMI) & (Nb[:, 1] > 0.5)
            onkalo = np.ones(len(P), np.float32)
            kuoppa_h = np.zeros(len(P), np.float32)
            if ylapinta.any():
                hh, gx, gz = kuoppa_ja_hanta(P[ylapinta, 0], P[ylapinta, 2])
                kuoppa_h[ylapinta] = hh
                Nn = np.stack([-gx, np.ones_like(gx), -gz], 1)
                Nb[ylapinta] = Nn / np.linalg.norm(Nn, axis=1, keepdims=True)
                onkalo[ylapinta] = np.clip(1 + hh / KUOPPA['syv'] * 0.18, 0.7, 1.05)
            # rounded front edge of the seat snow: at the edge the normal turns 45° toward each surface
            reuna = 0.03
            lumilaatikko = idx[penkki] == ISTUINLUMI
            yla = lumilaatikko & (Nb[:, 1] > 0.3)
            if yla.any():
                t = np.clip((P[yla, 2] - (B1[ISTUINLUMI][2] - reuna)) / reuna, 0, 1) * (np.pi / 4)
                ny, nz = Nb[yla, 1].copy(), Nb[yla, 2].copy()
                Nb[yla, 1] = ny * np.cos(t) - nz * np.sin(t)
                Nb[yla, 2] = ny * np.sin(t) + nz * np.cos(t)
            etu = lumilaatikko & (Nb[:, 2] > 0.5) & ~yla
            if etu.any():
                t = np.clip((P[etu, 1] - (B1[ISTUINLUMI][1] - reuna)) / reuna, 0, 1) * (np.pi / 4)
                Nb[etu] = np.stack([np.zeros_like(t), np.sin(t), np.cos(t)], 1)
            # softer vertical faces for the snow: wrap lighting (`kaari`)
            kaari = np.where(m == LUMI, 0.35, 0.0)
            c, E = valaise(P, Nb, alb, kaari=0.0)
            if (m == LUMI).any():
                c2, _ = valaise(P[m == LUMI], Nb[m == LUMI], alb[m == LUMI], kaari=0.35)
                c[m == LUMI] = c2
            ymp = alb * (TAIVAS * (0.55 + 0.45 * Nb[:, 1:2]) + POMPPU * (0.5 - 0.5 * Nb[:, 1:2]))
            lamppuosa = c - ymp
            sv = varjo(P, Nb)
            c = lamppuosa * sv[:, None] + ymp
            # packed, slightly frozen snow at the bottom of the hollow glints in the lamp light
            if ylapinta.any():
                Pp, Np = P[ylapinta], Nb[ylapinta]
                Ld = LAMPPU - Pp; Ld /= np.linalg.norm(Ld, axis=1, keepdims=True)
                Vd = CAM - Pp; Vd /= np.linalg.norm(Vd, axis=1, keepdims=True)
                Hv = Ld + Vd; Hv /= np.linalg.norm(Hv, axis=1, keepdims=True)
                syvyys_osuus = np.clip(-kuoppa_h[ylapinta] / KUOPPA['syv'], 0, 1)
                kiilto = np.clip((Np * Hv).sum(1), 0, 1) ** 40 * syvyys_osuus ** 2 * float(A.get('kiilto', 0.5))
                c[ylapinta] += (kiilto * sv[ylapinta])[:, None] * LAMPUN_VARI
            c *= onkalo[:, None]
            # wood grain: light streaking in the x direction
            puu = m == PUU
            if puu.any():
                syy = 0.88 + 0.12 * arvokohina(P[puu, 0] * 1.0, P[puu, 1] * 40 + P[puu, 2] * 40, 0.05, 7)
                c[puu] *= syy[:, None]
            sumu = 1 - np.exp(-t_b[penkki] / 9.0)
            c = c * (1 - sumu[:, None]) + SUMU * sumu[:, None]
            vari[penkki] = c
            a_ = ALUE['kartta'][y0:y1].reshape(-1)
            a_[penkki] = np.where(kuoppa_h < -0.003, 5, np.where(m == PUU, 2, np.where(m == METALLI, 3, 4)))
            v_ = ALUE['valo'][y0:y1].reshape(-1); v_[penkki] = E * sv
            p_ = ALUE['painauma'][y0:y1].reshape(-1); p_[penkki] = kuoppa_h
            syv = syvyys[y0:y1].reshape(-1); syv[penkki] = t_b[penkki]
        # --- sky and a distant row of trees
        if taivas.any():
            Dt = D[taivas]
            korkeus = Dt[:, 1]
            atsi = np.arctan2(Dt[:, 0], -Dt[:, 2])
            puut = 0.018 + 0.022 * arvokohina(atsi * 40, np.zeros_like(atsi), 1.0, 9) + 0.012 * arvokohina(atsi * 160, np.zeros_like(atsi), 1.0, 10)
            sky = SUMU * (1.0 - np.clip(korkeus * 6, 0, 0.6))[:, None]
            puussa = korkeus < puut
            sky[puussa] = SUMU * 0.45
            vari[taivas] = sky
        kuva[y0:y1] = vari.reshape(y1 - y0, RW, 3)
    return kuva, syvyys


def lumisade(kuva, maara=int(A.get('hiutaleet', 260))):
    """Drifting flakes: 3D position, size by depth, brightness from the lamp light."""
    k = kuva.copy()
    P = np.stack([rng.uniform(-2.0, 2.4, maara), rng.uniform(0.05, 2.4, maara), rng.uniform(-2.0, 2.6, maara)], 1)
    for p in P:
        v = p - CAM
        z = v @ f
        if z < 0.4:
            continue
        sx = (v @ r) / z / np.tan(FOV / 2)
        sy = (v @ u) / z / (np.tan(FOV / 2) * RH / RW)
        px, py = (sx + 1) / 2 * RW, (1 - sy) / 2 * RH
        if not (0 <= px < RW and 0 <= py < RH):
            continue
        Lv = LAMPPU - p; d2 = Lv @ Lv
        alas = Lv[1] / np.sqrt(d2)
        kartio = np.clip((alas - KARTIO[1]) / (KARTIO[0] - KARTIO[1]), 0, 1) ** 1.5
        kirkas = 0.02 + TEHO * kartio / d2 * 0.30
        sade = max(0.9 * SS, 0.0045 / z * RW / 2 / np.tan(FOV / 2))
        x0, x1 = int(max(0, px - sade * 3)), int(min(RW, px + sade * 3 + 1))
        y0, y1 = int(max(0, py - sade * 3)), int(min(RH, py + sade * 3 + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        a = np.exp(-(((xx - px) / sade) ** 2 + ((yy - py) / (sade * 1.6)) ** 2))
        a *= 0.85 if z > 1.2 else 0.5                        # the nearest flakes are out of focus
        k[y0:y1, x0:x1] += (a[..., None] * kirkas * np.array([1.0, 0.95, 0.9]))
    return k


def savy(x):
    x = x * float(A.get('valotus', 1.45))
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)   # ACES fit: contrast in the shadows
    y = np.clip(y, 0, 1) ** (1 / 2.2)
    return y


if __name__ == '__main__':
    kuva, _ = renderoi()
    if 'opas' in A:                     # underdrawing for the hand drawing (ss=1)
        import json
        vari = (np.clip(savy(kuva), 0, 1) * 255).astype(np.uint8)
        np.savez_compressed(A['opas'], vari=vari, alue=ALUE['kartta'], valo=ALUE['valo'], painauma=ALUE['painauma'])
        jaljet = [dict(x=float(projisoi([j[0], 0, j[1]])[0]), y=float(projisoi([j[0], 0, j[1]])[1]),
                       ex=float(projisoi([j[0] + 0.05 * j[2], 0, j[1] + 0.05 * j[3]])[0]), ey=float(projisoi([j[0] + 0.05 * j[2], 0, j[1] + 0.05 * j[3]])[1]))
                  for j in JALJET]
        y_ = B1[ISTUINLUMI][1]
        kuoppa = [list(map(float, projisoi([KUOPPA['x'] + KUOPPA['rx'] * np.cos(a), y_, KUOPPA['z'] + KUOPPA['rz'] * np.sin(a)])))
                  for a in np.linspace(0, 2 * np.pi, 49)]
        lamppu = list(map(float, projisoi([LAMPPU[0], 0.0, LAMPPU[2]])))
        json.dump(dict(jaljet=jaljet, kuoppa=kuoppa, lampun_alla=lamppu), open(A['opas'][:-4] + '.json', 'w'), indent=1)
        print('pohjapiirros', A['opas'])
        sys.exit(0)
    kuva = lumisade(kuva)
    img = savy(kuva)
    # vignette
    yy, xx = np.mgrid[0:RH, 0:RW]
    rr = np.sqrt(((xx - RW / 2) / (RW / 2)) ** 2 + ((yy - RH * 0.55) / (RH / 2)) ** 2)
    img *= (1 - 0.28 * np.clip(rr - 0.55, 0, 1) ** 1.5)[..., None]
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + rng.random(img.shape) * 0.999).astype(np.uint8))
    im = im.resize((W, H), Image.LANCZOS)
    im.save(A.get('ulos', 'kansi.png'))
    print('valmis', A.get('ulos', 'kansi.png'), im.size)
