"""Näkyvä muisti (Visible Memory): a den under the roots of an old oak, in the manner of a storybook (v2).

A winter night on the seashore. Under the ground, the den is shown in cross
section: Kettu (the fox) sleeps curled up on a bed of moss, with the tail
wrapped around Lumi. Lumi is awake and looks along the tunnel up to the sky,
where Saturn is. In the snow, a hare's tracks come to the mouth of the den
and go back. The roots of the oak hold the den like hands.

The picture is a vignette, as in old storybook pictures, but the edges come
from the colors themselves: sky, sea, snow, earth and tree each end in
their own place, with a wet edge, a dry brush or a soft fade.

Run: python3 pesa_satu.py out.jpg
"""
import sys
import numpy as np
import cv2
import akvarelli as ak
from akvarelli import (W, H, X, Y, noise, new_canvas, catmull, poly_mask, ellipse_mask, thick_path_poly,
                       glaze, gouache, screen, Ink, hatch, Pencil, save, PAPER_HF, wobble)
import hahmot as hm
from hahmot_testi import paint_animals

rng = np.random.default_rng(17)

# ---------------------------------------------------------------- reach of the washes
# Johannes's suggestion on 17 Nov: the edges come from where each color spreads.
# Each color (sky, far shore, sea, snow, earth, tree) ends in its own
# place, and the ends of the horizontal strokes are left at different lengths.


def edge_line(n, base, low_amp, low_scale, row, row_amp, seed):
    """The end of a wash as a distance from the picture edge: a slow wave and the ends of the strokes."""
    r = np.random.default_rng(seed)
    nz = noise(low_scale, seed + 1)
    low = nz[:, 11] if n == H else nz[11, :]
    out = np.zeros(n, np.float32)
    if row > 0 and row_amp > 0:
        i = 0
        while i < n:
            h = max(4, int(r.uniform(0.55, 1.45) * row))
            out[i:i + h] = r.uniform(-1, 1) * row_amp
            i += h
        out = cv2.GaussianBlur(out.reshape(1, -1), (0, 0), max(1.5, row * 0.10)).ravel()
    return (base + low_amp * low + out).astype(np.float32)


def reach_field(L=None, R=None, T=None, B=None, seed=0):
    """Reach field: the distance to the end of the wash (L, R, T and B are given as distances from the picture edges)."""
    F = np.full((H, W), 4000.0, np.float32)
    if L is not None:
        F = np.minimum(F, X - L[:, None])
    if R is not None:
        F = np.minimum(F, (W - R)[:, None] - X)
    if T is not None:
        F = np.minimum(F, Y - T[None, :])
    if B is not None:
        F = np.minimum(F, (H - B)[None, :] - Y)
    F += 3.5 * noise(6, seed) + 6 * noise(20, seed + 1) + 11 * noise(70, seed + 2, octaves=2)
    return F


def thick_path_poly_w(c, w):
    """Like thick_path_poly, but the width is given for each point."""
    c = np.asarray(c, np.float64)
    d = np.gradient(c, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    hw = np.asarray(w, np.float64)[:, None] / 2
    return np.vstack([c + nrm * hw, (c - nrm * hw)[::-1]])


def splatter(cv, F, region, col, n, seed):
    """A few splatters outside the edge of the wash."""
    r = np.random.default_rng(seed)
    cand = np.argwhere((F < -8) & (F > -75) & (region > 0.5))
    if len(cand) == 0:
        return
    sm = np.zeros((H, W), np.float32)
    for k in r.choice(len(cand), n, replace=False):
        y, x = cand[k]
        rad = r.choice([1.2, 1.6, 2.2, 3.0, 4.2, 6.0], p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
        sm = np.maximum(sm, ellipse_mask(x, y, rad * r.uniform(0.9, 1.5), rad, r.uniform(0, 180)))
    glaze(cv, sm, col, a=1.0, edge=0.9, edge_w=1.5, gran=0.3, bloom=0, seed=seed, wob=0.6, wob_scale=3)

# ---------------------------------------------------------------- terrain
GCTRL = [(-40, 640), (200, 632), (420, 622), (620, 612), (800, 604), (960, 602), (1120, 606), (1320, 614),
         (1520, 621), (1720, 627), (1950, 633), (2200, 639), (2440, 645)]
gl = catmull(GCTRL, closed=False, n=30)
o = np.argsort(gl[:, 0])
gx, gy = gl[o, 0], gl[o, 1]


def g(x):
    return np.interp(x, gx, gy)


HOR = 455.0
xs = np.arange(W, dtype=np.float32)
gline = g(xs).astype(np.float32)
shore = (536 + 9 * noise(120, 7)[300, :] + 3 * noise(25, 8)[300, :]).astype(np.float32)
above_cut = (Y < gline[None, :]).astype(np.float32)
below_cut = 1 - above_cut
sky_m = (Y < HOR).astype(np.float32)

fs = np.full(W, HOR, np.float32)
hill = HOR - (30 + 10 * noise(160, 21)[400, :]) * np.clip((1100 - xs) / 280, 0, 1)
spruce = np.zeros(W, np.float32)
for cx in rng.uniform(0, 1080, 70):
    hh = rng.uniform(7, 22)
    ww = rng.uniform(5, 11)
    spruce = np.maximum(spruce, np.clip(hh * (1 - np.abs(xs - cx) / ww), 0, None))
fs = np.minimum(fs, hill - spruce * (xs < 1100))
island = HOR - 12 * np.exp(-((xs - 1840) / 85) ** 2) - 4 * np.exp(-((xs - 1750) / 35) ** 2)
fs = np.minimum(fs, island)
farshore_m = ((Y >= fs[None, :]) & (Y < HOR + 2)).astype(np.float32)
sea_m = ((Y >= HOR) & (Y < shore[None, :])).astype(np.float32) * above_cut
snowfield_m = ((Y >= shore[None, :]) & (Y < gline[None, :])).astype(np.float32)

# ---------------------------------------------------------------- tree
trunk_pts = [(700, 628), (770, 602), (812, 560), (836, 480), (846, 390), (842, 300), (834, 200), (826, 100), (822, -40),
             (990, -40), (988, 100), (994, 200), (1006, 300), (1024, 390), (1050, 470), (1090, 540), (1150, 590),
             (1230, 612), (1150, 640), (800, 646)]
trunk_m = poly_mask(catmull(trunk_pts, closed=True, n=10)) * above_cut
# The branches start inside the trunk, so no sky is left between trunk and branch.
# At the joint the branch widens a little (branch collar). The upper left branch
# is in front of the trunk, on the viewer's side (Johannes's suggestion on 17 Nov).
limbs = [
    ([(910, 256), (850, 230), (780, 200), (700, 182), (640, 150), (560, 128), (500, 96), (420, 80), (350, 40), (270, 20), (200, -20)], 80, 18, True),
    ([(934, 222), (990, 190), (1060, 150), (1150, 130), (1210, 96), (1300, 80), (1370, 40), (1440, 20), (1520, -30)], 76, 20, False),
    ([(952, 347), (1030, 330), (1120, 310), (1190, 318), (1280, 290), (1370, 282), (1450, 258), (1540, 256), (1620, 236), (1700, 232), (1780, 214)], 56, 7, False),
    ([(906, 382), (842, 370), (770, 356), (690, 362), (610, 330), (530, 322), (450, 296), (370, 290), (290, 256), (210, 250)], 50, 7, False),
]
trunk_in = cv2.distanceTransform((trunk_m > 0.5).astype(np.uint8), cv2.DIST_L2, 3)  # depth inside the trunk
limb_m = np.zeros((H, W), np.float32)
front_m = np.zeros((H, W), np.float32)
limb_paths = []
for pts, w0, w1, front in limbs:
    c = catmull(pts, closed=False, n=12)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    inside = np.array([trunk_m[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))] > 0.5 for p in c])
    ie = int(np.argmin(inside)) if not inside.all() else 0
    se = s[ie]
    u = np.clip((s - se) / (s[-1] - se), 0, 1)
    w_lin = w0 * (1 - u) + w1 * u
    w = w_lin * (1 + 0.32 * np.exp(-np.clip(s - se, 0, None) / 34.0))
    limb_paths.append((c, w, w_lin, front))
    lm = poly_mask(thick_path_poly_w(c, w))
    limb_m = np.maximum(limb_m, lm)
    if front:
        front_m = np.maximum(front_m, lm)
# the visible part of the front branch on top of the trunk: about 36 px in from the trunk edge, then a fade
front_over = front_m * (trunk_m > 0.5) * np.clip((44 - trunk_in) / 12, 0, 1)
twigs = []


def grow(p, ang, length, w, depth, seed):
    r = np.random.default_rng(seed)
    pts = [np.array(p, float)]
    a = ang
    steps = max(3, int(length / 16))
    for i in range(steps):
        a += r.normal(0, 0.3)
        pts.append(pts[-1] + length / steps * np.array([np.cos(a), np.sin(a)]))
    twigs.append((np.array(pts), w))
    if depth > 0:
        for k in range(r.integers(2, 4)):
            j = r.integers(1, len(pts) - 1)
            na = a + r.choice([-1, 1]) * r.uniform(0.35, 0.9)
            grow(pts[j], na, length * r.uniform(0.45, 0.7), w * 0.6, depth - 1, seed * 7 + k + 1)


for (c, w, w_lin, front), seedb in zip(limb_paths, [3, 5, 9, 13]):
    n = len(c)
    for k, fi in enumerate([0.22, 0.38, 0.52, 0.66, 0.8, 0.97]):
        i = min(n - 2, 12 + int(fi * (n - 12)))  # 12 = the stretch added inside the trunk
        d = c[i + 1] - c[i]
        base = np.arctan2(d[1], d[0])
        side = -1 if k % 2 == 0 else 1
        grow(c[i], base + side * 0.55 - 0.25, 230 * (1.1 - fi * 0.5), max(2.5, w_lin[i] * 0.32), 3, seedb * 31 + k)
twig_m = np.zeros((H, W), np.float32)
for pts, w in twigs:
    twig_m = np.maximum(twig_m, poly_mask(thick_path_poly(pts, w, max(1.2, w * 0.35))))
tree_up_m = np.maximum(trunk_m, limb_m)

# surface roots: the right root arches over the mouth of the den
rootD_c = catmull([(1150, 606), (1250, 598), (1330, 600), (1400, 612), (1440, 628)], closed=False, n=12)
rootD_m = poly_mask(thick_path_poly(rootD_c, 40, 14)) * (Y < gline[None, :] + 1)
rootL_c = catmull([(720, 624), (600, 632), (500, 644)], closed=False, n=10)
rootL_m = poly_mask(thick_path_poly(rootL_c, 42, 16))

# ---------------------------------------------------------------- den and tunnel
DEN = [(646, 980), (668, 878), (736, 808), (856, 764), (1008, 746), (1160, 750), (1290, 778), (1378, 830), (1440, 886),
       (1492, 956), (1510, 1030), (1494, 1106), (1424, 1168), (1262, 1204), (1060, 1216), (868, 1210), (728, 1182), (658, 1102)]
den_m = poly_mask(catmull(DEN, closed=True, n=12))
tun_c = catmull([(1418, 872), (1474, 830), (1518, 782), (1550, 734), (1576, 688), (1602, 648), (1624, 618), (1638, 604)], closed=False, n=14)
tun_poly = thick_path_poly(tun_c, 96, 78)
tun_m = poly_mask(tun_poly) * (Y > gline[None, :] + 1)
tun_m = wobble(tun_m, 3, 30, 811)
hollow_m = np.clip(np.maximum(den_m, tun_m), 0, 1)
MOUTH = np.array([1632, 622.0])
LANTERN = (818, 876)

# ---------------------------------------------------------------- roots in the ground
den_c = catmull(DEN, closed=True, n=12)
dd = np.gradient(den_c, axis=0)
dd /= (np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9)
out_n = np.stack([dd[:, 1], -dd[:, 0]], 1)
# outward-pointing normal: check the direction from the center point
ctr = den_c.mean(0)
if np.mean(np.sum((den_c - ctr) * out_n, 1)) < 0:
    out_n = -out_n
ring_c = den_c + out_n * 34
ang = np.degrees(np.arctan2(ring_c[:, 1] - ctr[1], ring_c[:, 0] - ctr[0]))
# left cradle root: from the upper left down and under the floor; right: from the lower right under the floor
left_idx = [i for i in range(len(ring_c)) if (ang[i] > 95 or ang[i] < -150)]
right_idx = [i for i in range(len(ring_c)) if (10 < ang[i] < 95)]


def ordered(idx):
    idx = sorted(idx, key=lambda i: (ang[i] % 360))
    return ring_c[idx]


cradleL = ordered(left_idx)
cradleL = cradleL[::-1] if cradleL[0][1] > cradleL[-1][1] else cradleL
cradleR = ordered(right_idx)
roots_ground = [
    ([(780, 632), (730, 700)] + [tuple(p) for p in cradleL[::4]], 64, 14),
    # the right cradle root goes around the right side of the den, under the floor, and ends before the tip of the left root
    # (earlier the list was reversed, and the root folded over itself at the bottom)
    ([(1500, 1000)] + [tuple(p) for p in cradleR[::4] if p[0] > 1105], 44, 12),
    ([(900, 634), (960, 690), (1060, 716), (1180, 724), (1300, 734), (1400, 768)], 34, 12),
    ([(620, 636), (520, 720), (430, 820), (340, 930), (250, 1030)], 40, 8),
    ([(1840, 642), (1910, 720), (1990, 800), (2080, 860), (2180, 900)], 28, 6),
]
root_paths = []
roots_m = np.zeros((H, W), np.float32)
for pts, w0, w1 in roots_ground:
    c = catmull(pts, closed=False, n=12)
    root_paths.append((c, w0, w1))
    roots_m = np.maximum(roots_m, poly_mask(thick_path_poly(c, w0, w1)))
    roots_m = np.maximum(roots_m, ellipse_mask(c[-1][0], c[-1][1], w1 / 2, w1 / 2))  # round tip
side_roots = []
for (c, w0, w1), sd in zip(root_paths, [41, 43, 47, 53, 59, 61]):
    r = np.random.default_rng(sd)
    for k in range(3):
        i = r.integers(2, len(c) - 3)
        d = c[i + 1] - c[i]
        a = np.arctan2(d[1], d[0]) + r.choice([-1, 1]) * r.uniform(0.5, 1.1)
        p = [c[i]]
        for s in range(9):
            a += r.normal(0, 0.3)
            p.append(p[-1] + 13 * np.array([np.cos(a), np.sin(a)]))
        side_roots.append(np.array(p))
for p in side_roots:
    roots_m = np.maximum(roots_m, poly_mask(thick_path_poly(p, 7, 1.5)))
roots_m = roots_m * (Y > gline[None, :] + 30)
# The roots of the trunk continue through the snow layer up to the cut line (Johannes's remark on 17 Nov).
# A stub in the root's own direction is added to the start of the root and clipped to the snow layer.
ROOT_UP = (0, 2)  # the roots under the trunk; the ones farther away start below the snow layer
stubs = []
stub_m = np.zeros((H, W), np.float32)
for k in ROOT_UP:
    c, w0, w1 = root_paths[k]
    d = c[4] - c[0]
    d /= np.linalg.norm(d)
    seg = np.array([c[0] - d * 50, c[0], c[0] + d * 14])
    stubs.append((seg, w0))
    stub_m = np.maximum(stub_m, poly_mask(thick_path_poly(seg, w0, w0)))
stub_m = stub_m * (Y > gline[None, :] + 1) * (Y < gline[None, :] + 40)
roots_m = np.maximum(roots_m, stub_m)
roots_vis = roots_m * (1 - hollow_m)

CUTS = [(330, 772, 14), (1830, 980, 10), (1960, 1100, 15), (430, 1210, 12), (2150, 1060, 9), (205, 712, 8),
        (1230, 1300, 9), (520, 950, 8)]
STONES = []
r = np.random.default_rng(77)
for i in range(34):
    x = r.uniform(80, W - 80)
    yy = r.uniform(720, H - 60)
    if hollow_m[int(min(yy, H - 1)), int(min(x, W - 1))] > 0 or roots_m[int(min(yy, H - 1)), int(min(x, W - 1))] > 0:
        continue
    STONES.append((x, yy, r.uniform(7, 18), r.uniform(5, 12), r.uniform(-30, 30)))
BOULDER = [(1700, 1080), (1738, 1036), (1800, 1018), (1880, 1016), (1950, 1032), (1996, 1066), (2004, 1110), (1972, 1140),
           (1890, 1152), (1790, 1150), (1726, 1128)]
BOULDER2 = []
boulder_m = poly_mask(catmull(BOULDER, closed=True, n=6))

# ---------------------------------------------------------------- reach of each color
# (distances from the picture edges: base, slow wave, stroke height and variation)
F_SKY = reach_field(L=edge_line(H, 62, 24, 220, 30, 40, 1101), R=edge_line(H, 104, 28, 240, 26, 50, 1102),
                    T=edge_line(W, 36, 14, 300, 0, 0, 1103) - 110 * np.exp(-((xs - 906) / 175) ** 2), seed=1104)
# (at the trunk the color rises all the way to the top edge, so the oak continues out of the picture)
F_FAR = reach_field(L=edge_line(H, 116, 16, 200, 9, 26, 1111), R=edge_line(H, 150, 20, 200, 9, 30, 1112), seed=1113)
F_SEA = reach_field(L=edge_line(H, 22, 8, 160, 11, 16, 1121), R=edge_line(H, 28, 10, 160, 12, 20, 1122), seed=1123)
F_SNOW = reach_field(L=edge_line(H, 96, 18, 200, 15, 28, 1131), R=edge_line(H, 80, 18, 200, 15, 26, 1132), seed=1133)
F_CUT = reach_field(L=edge_line(H, 70, 12, 200, 0, 0, 1141), R=edge_line(H, 62, 12, 200, 0, 0, 1142), seed=1143)
F_EARTH = reach_field(L=edge_line(H, 54, 26, 260, 44, 34, 1151), R=edge_line(H, 50, 24, 260, 50, 40, 1152),
                      B=edge_line(W, 46, 24, 280, 0, 0, 1153), seed=1154)
F_TREE = F_SKY + 4  # at the top the tree ends roughly where the sky does
# the sky is painted around the tree only where the tree color is sure to be; higher up the sky covers the end of the tree
tree_sure = np.clip(tree_up_m, 0, 1) * np.clip((F_TREE - 24) / 8, 0, 1)
STONES = [s for s in STONES if F_EARTH[int(min(s[1], H - 1)), int(min(s[0], W - 1))] > 30]
CUTS = [c_ for c_ in CUTS if F_EARTH[int(min(c_[1], H - 1)), int(min(c_[0], W - 1))] > 30]
around_tree = 1 - np.clip(tree_up_m, 0, 1)  # the sea and the far shore are painted around the trunk

# ---------------------------------------------------------------- light
dl = np.hypot(X - LANTERN[0], Y - LANTERN[1])
light_den = np.exp(-dl / 360.0) * hollow_m
light_den = cv2.GaussianBlur(light_den, (0, 0), 6)
dm = np.hypot(X - MOUTH[0], (Y - MOUTH[1]) * 1.9)
light_mouth = np.exp(-dm / 140.0) * (Y < gline[None, :] + 4)


def stars_list():
    rs = np.random.default_rng(3)
    out = []
    for i in range(280):
        x, y = rs.uniform(0, W), rs.uniform(0, HOR - 18)
        rr = rs.choice([0.9, 1.1, 1.4, 1.8, 2.6], p=[0.35, 0.3, 0.2, 0.1, 0.05])
        out.append((x, y, rr))
    return out


def render(out):
    cv = new_canvas()
    stars = stars_list()
    star_m = np.zeros((H, W), np.float32)
    for (x, y, rr) in stars:
        cv2.circle(star_m, (int(x * 4), int(y * 4)), int(rr * 4), 1.0, -1, cv2.LINE_AA, 2)
    keep = (1 - star_m) * (1 - tree_sure)  # the sky is painted around the tree

    # ---------- sky
    ak.set_reach(F_SKY)
    s = np.clip(Y / HOR, 0, 1) ** 1.8
    top1, bot1 = np.array([0.46, 0.56, 0.80]), np.array([0.78, 0.84, 0.93])
    top2, bot2 = np.array([0.42, 0.42, 0.76]), np.array([0.88, 0.88, 0.96])
    col1 = top1[None, None, :] * (1 - s[..., None]) + bot1[None, None, :] * s[..., None]
    col2 = top2[None, None, :] * (1 - s[..., None]) + bot2[None, None, :] * s[..., None]
    glaze(cv, sky_m * keep, col1, a=1.0, edge=0.25, edge_w=6, gran=0.25, bloom=0.22, bloom_scale=160, seed=101, wob=0)
    glaze(cv, sky_m * keep, col2, a=1.0, edge=0.2, edge_w=6, gran=0.3, bloom=0.3, bloom_scale=90, seed=102, wob=0)
    topdark = np.clip(1 - Y / 360, 0, 1) ** 1.3 * sky_m * keep
    glaze(cv, topdark, (0.55, 0.58, 0.70), a=1.0, edge=0, gran=0.3, bloom=0.35, bloom_scale=220, seed=103, wob=0)

    # ---------- far shore, sea, shore
    ak.set_reach(F_FAR)
    glaze(cv, farshore_m * around_tree, (0.38, 0.44, 0.50), a=1.0, edge=0.4, edge_w=3, gran=0.4, bloom=0.2, seed=111, wob=1.2, wob_scale=10)
    glaze(cv, farshore_m * around_tree, (0.58, 0.64, 0.64), a=0.9, edge=0.3, gran=0.5, bloom=0.3, seed=112, wob=1.2, wob_scale=10)
    ak.set_reach(F_SEA)
    sm = np.clip((Y - HOR) / 80, 0, 1)
    seacol = np.array([0.72, 0.77, 0.88])[None, None, :] * (1 - sm[..., None]) + np.array([0.84, 0.88, 0.95])[None, None, :] * sm[..., None]
    glaze(cv, sea_m * around_tree, seacol, a=1.0, edge=0.2, gran=0.3, bloom=0.25, bloom_scale=120, seed=121, wob=0)
    streaks = np.clip(noise(30, 122) * 1.6, 0, 1) * sea_m
    streaks = cv2.GaussianBlur(cv2.resize(cv2.resize(streaks, (W // 10, H)), (W, H)), (0, 0), 2) * around_tree
    glaze(cv, streaks, (0.84, 0.87, 0.94), a=0.5, edge=0, gran=0.3, bloom=0, seed=123, wob=0)
    shore_band = ((Y >= shore[None, :] - 4) & (Y < shore[None, :] + 4)).astype(np.float32) * above_cut * around_tree
    glaze(cv, shore_band * (noise(9, 124) > 0.25), (0.66, 0.68, 0.76), a=0.7, edge=0.2, gran=0.5, seed=125, wob=1.2, wob_scale=5)

    # ---------- snowfield
    ak.set_reach(F_SNOW)
    glaze(cv, snowfield_m * around_tree, (0.88, 0.91, 0.98), a=0.7, edge=0.2, gran=0.2, bloom=0.4, bloom_scale=90, seed=131, wob=0)
    shadow_tree = np.clip(1 - np.abs((X - 1260 - (Y - 560) * 2.4) / 170), 0, 1) * snowfield_m * (X > 1000) * around_tree
    glaze(cv, cv2.GaussianBlur(shadow_tree, (0, 0), 14), (0.80, 0.84, 0.95), a=0.75, edge=0, gran=0.3, bloom=0.2, seed=132, wob=0)
    glaze(cv, light_mouth * snowfield_m, (1.0, 0.80, 0.48), a=1.0, edge=0, gran=0.15, bloom=0.15, seed=133, wob=0)
    glaze(cv, light_mouth ** 2 * snowfield_m, (1.0, 0.86, 0.60), a=0.9, edge=0, gran=0.1, bloom=0, seed=1331, wob=0)

    # the hare's tracks: they come from the right to the mouth and go back farther away
    track_m = np.zeros((H, W), np.float32)
    path_in = catmull([(2300, 600), (2140, 606), (1990, 612), (1850, 616), (1712, 612)], closed=False, n=40)
    path_out = catmull([(1740, 598), (1860, 584), (2010, 574), (2180, 566), (2300, 562)], closed=False, n=40)
    for pth, direction in [(path_in, -1), (path_out, 1)]:
        seg = np.linalg.norm(np.diff(pth, axis=0), axis=1)
        sacc = np.concatenate([[0], np.cumsum(seg)])
        dist = 30.0
        while dist < sacc[-1] - 10:
            x = np.interp(dist, sacc, pth[:, 0])
            y = np.interp(dist, sacc, pth[:, 1])
            sc = 0.5 + 0.5 * np.clip((y - 560) / 60, 0, 1)
            dx = direction * sc
            for oy in (-4.6 * sc, 4.6 * sc):
                track_m = np.maximum(track_m, ellipse_mask(x + dx * 18, y + oy, 10 * sc, 3.2 * sc, 0))
            track_m = np.maximum(track_m, ellipse_mask(x - dx * 4, y + 1, 3.4 * sc, 2.6 * sc, 0))
            track_m = np.maximum(track_m, ellipse_mask(x - dx * 15, y - 1, 3.4 * sc, 2.6 * sc, 0))
            dist += 118 * sc
    glaze(cv, track_m, (0.74, 0.79, 0.93), a=0.7, edge=0.5, edge_w=1.5, gran=0.2, bloom=0, seed=135, wob=0.5, wob_scale=3)

    # a mossy erratic boulder on the snowfield
    BOULD = [(262, 624), (268, 584), (304, 552), (362, 538), (426, 548), (464, 584), (468, 622)]
    bould_m = poly_mask(catmull(BOULD, closed=True, n=10)) * above_cut
    glaze(cv, bould_m, (0.66, 0.66, 0.70), a=1.0, edge=0.5, gran=0.6, bloom=0.3, seed=141, wob=1.5, wob_scale=10)
    moss_side = bould_m * (Y > 582) * (noise(14, 142) > -0.25)
    glaze(cv, moss_side, (0.64, 0.76, 0.42), a=0.85, edge=0.4, gran=0.6, bloom=0.3, seed=143, wob=2, wob_scale=8)
    snowcap = bould_m * (Y < 568 + 8 * noise(20, 144))
    gouache(cv, cv2.GaussianBlur(snowcap, (0, 0), 1.2), (0.95, 0.95, 0.97), op=0.9)


    # ---------- cross section: snow layer, humus, earth (pale washes, the ink makes the depth)
    snowband = ((Y >= gline[None, :]) & (Y < gline[None, :] + 30 + 6 * noise(40, 151))).astype(np.float32)
    hmask = 1 - hollow_m
    ak.set_reach(F_CUT)
    glaze(cv, snowband * hmask * (1 - stub_m), (0.86, 0.90, 0.97), a=0.5, edge=0.5, edge_w=3, gran=0.3, bloom=0.2, seed=152, wob=1)
    ak.set_reach(F_EARTH)
    soil = (Y >= gline[None, :] + 28).astype(np.float32)
    humus = ((Y >= gline[None, :] + 28) & (Y < gline[None, :] + 84 + 16 * noise(60, 153))).astype(np.float32)
    depth = np.clip((Y - gline[None, :] - 40) / 700, 0, 1)
    soilcol = np.array([0.76, 0.64, 0.50])[None, None, :] * (1 - depth[..., None]) + np.array([0.82, 0.70, 0.56])[None, None, :] * depth[..., None]
    # under the root stubs the same earth as elsewhere, so that the root is equally dark along its whole length
    glaze(cv, np.maximum(soil * hmask, stub_m), soilcol, a=1.0, edge=0.2, gran=0.5, bloom=0.4, bloom_scale=110, seed=154, wob=0)
    glaze(cv, np.maximum(soil * hmask, stub_m), (0.84, 0.76, 0.66), a=0.8, edge=0.2, gran=0.5, bloom=0.45, bloom_scale=60, seed=155, wob=0)
    glaze(cv, np.maximum(humus * hmask, stub_m), (0.66, 0.56, 0.48), a=1.0, edge=0.3, edge_w=3, gran=0.5, bloom=0.3, seed=156, wob=2, wob_scale=14)
    rim = cv2.GaussianBlur(hollow_m, (0, 0), 34) * hmask
    glaze(cv, rim * np.exp(-dl / 520), (1.0, 0.86, 0.64), a=0.55, edge=0, gran=0.2, bloom=0, seed=157, wob=0)

    st_m = np.zeros((H, W), np.float32)
    for (x, y, rx, ry, a) in STONES:
        st_m = np.maximum(st_m, ellipse_mask(x, y, rx, ry, a))
    st_m *= below_cut * hmask
    glaze(cv, st_m, (0.74, 0.74, 0.78), a=1.0, edge=0.6, edge_w=2, gran=0.6, bloom=0.4, bloom_scale=8, seed=161, wob=1.0, wob_scale=6)
    glaze(cv, boulder_m, (0.80, 0.79, 0.82), a=1.0, edge=0.5, gran=0.7, bloom=0.4, bloom_scale=40, seed=162, wob=1.5, wob_scale=12)
    glaze(cv, boulder_m * (noise(4, 163) > 0.55), (0.94, 0.76, 0.72), a=0.7, edge=0, gran=0.3, bloom=0, seed=164, wob=0)
    glaze(cv, boulder_m * np.clip((Y - 1050) / 100, 0, 1), (0.80, 0.80, 0.86), a=0.8, edge=0, gran=0.4, bloom=0, seed=165, wob=0)

    glaze(cv, roots_vis, (0.76, 0.66, 0.56), a=1.0, edge=0.6, edge_w=3, gran=0.5, bloom=0.3, bloom_scale=30, seed=171, wob=0.8, wob_scale=8)
    glaze(cv, roots_vis * np.clip(1 - light_den * 0.0, 0, 1), (0.86, 0.78, 0.70), a=0.7, edge=0.3, gran=0.4, bloom=0.4, seed=172, wob=0.8, wob_scale=8)
    cut_m = np.zeros((H, W), np.float32)
    for (x, y, rr) in CUTS:
        cut_m = np.maximum(cut_m, ellipse_mask(x, y, rr, rr * 0.85, 0))
    glaze(cv, cut_m * hmask, (0.86, 0.72, 0.56), a=1.0, edge=0.9, edge_w=2, gran=0.3, bloom=0, seed=173, wob=0.5, wob_scale=4)

    # ---------- inside of the den
    ak.set_reach(None)
    glaze(cv, hollow_m, (0.99, 0.88, 0.64), a=1.0, edge=0.8, edge_w=8, gran=0.35, bloom=0.3, bloom_scale=80, seed=181, wob=1.5, wob_scale=20)
    far = np.clip(1 - light_den * 1.5, 0, 1) * hollow_m
    glaze(cv, far, (0.84, 0.68, 0.48), a=1.0, edge=0.5, edge_w=10, gran=0.4, bloom=0.35, bloom_scale=60, seed=182, wob=1.5, wob_scale=20)
    far2 = np.clip(1 - light_den * 2.4, 0, 1) ** 1.5 * hollow_m
    glaze(cv, far2, (0.76, 0.62, 0.54), a=0.85, edge=0.3, gran=0.45, bloom=0.3, seed=183, wob=1.5, wob_scale=20)
    # shadow of the walls (inner edges)
    hb = (hollow_m > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(hb, cv2.DIST_L2, 3)
    wall = np.exp(-din / 26.0) * hollow_m
    glaze(cv, wall, (0.80, 0.66, 0.54), a=0.8, edge=0, gran=0.4, bloom=0.3, seed=185, wob=0)
    # the upper end of the tunnel cools into moonlight
    tpar = np.clip((900 - Y) / 280, 0, 1) * tun_m * (1 - den_m)
    glaze(cv, tpar, (0.82, 0.70, 0.60), a=0.9, edge=0.2, gran=0.35, bloom=0.2, seed=184, wob=0)
    glaze(cv, np.clip((780 - Y) / 150, 0, 1) * tun_m * (1 - den_m), (0.62, 0.62, 0.76), a=0.9, edge=0, gran=0.3, bloom=0.2, seed=186, wob=0)
    glaze(cv, np.clip((690 - Y) / 70, 0, 1) * tun_m * (1 - den_m), (0.66, 0.66, 0.80), a=0.8, edge=0, gran=0.3, bloom=0.1, seed=187, wob=0)

    m = hm.masks()
    fox_lumi = np.clip(m['fox_all'] + m['lumi_full'], 0, 1)
    # bed of moss (painted around the animals)
    bed_m = ellipse_mask(1080, 1188, 360, 34, 0) * den_m * (1 - fox_lumi)
    glaze(cv, bed_m, (0.70, 0.78, 0.46), a=1.0, edge=0.6, edge_w=4, gran=0.6, bloom=0.4, bloom_scale=20, seed=191, wob=3, wob_scale=10)
    glaze(cv, bed_m * (noise(9, 192) > 0.2), (0.60, 0.70, 0.40), a=0.9, edge=0.3, gran=0.6, bloom=0, seed=193, wob=0)
    leaf_m = np.zeros((H, W), np.float32)
    rl = np.random.default_rng(55)
    for i in range(30):
        x = rl.uniform(700, 1480)
        y = rl.uniform(1160, 1205)
        if den_m[int(y), int(x)] < 0.5:
            continue
        leaf_m = np.maximum(leaf_m, ellipse_mask(x, y, rl.uniform(9, 15), rl.uniform(4, 7), rl.uniform(0, 180)))
    glaze(cv, leaf_m * (1 - fox_lumi), (0.90, 0.64, 0.38), a=1.0, edge=0.7, edge_w=1.5, gran=0.4, bloom=0, seed=194, wob=0.6, wob_scale=3)

    niche = [(690, 990), (698, 954), (790, 948), (798, 984), (790, 1018), (698, 1022)]
    niche_m = poly_mask(catmull(niche, closed=True, n=8))
    glaze(cv, niche_m, (0.70, 0.56, 0.44), a=1.0, edge=0.6, edge_w=3, gran=0.5, bloom=0, seed=201, wob=1, wob_scale=6)
    books = [(708, 978, 12, 40, (0.76, 0.44, 0.38)), (722, 982, 10, 36, (0.52, 0.62, 0.50)),
             (734, 976, 13, 42, (0.52, 0.56, 0.72)), (748, 986, 9, 32, (0.84, 0.72, 0.46))]
    for (bx, by, bw, bh, bc) in books:
        bm = poly_mask([(bx, by + bh / 2), (bx, by - bh / 2), (bx + bw, by - bh / 2), (bx + bw, by + bh / 2)])
        glaze(cv, bm, bc, a=1.0, edge=0.5, edge_w=2, gran=0.4, bloom=0, seed=int(bx), wob=0.4, wob_scale=4)
    glaze(cv, ellipse_mask(775, 1006, 14, 6, -8), (0.97, 0.93, 0.82), a=0.6, edge=0.6, gran=0.2, bloom=0, seed=209, wob=0)

    # ---------- tree
    ak.set_reach(F_TREE)
    tree_all = np.maximum(tree_up_m, np.maximum(rootD_m, rootL_m) * above_cut)
    glaze(cv, tree_all, (0.64, 0.56, 0.48), a=1.0, edge=0.5, edge_w=4, gran=0.5, bloom=0.3, bloom_scale=50, seed=211, wob=1.0, wob_scale=10)
    glaze(cv, tree_all, (0.70, 0.70, 0.78), a=0.9, edge=0.3, gran=0.5, bloom=0.3, seed=2111, wob=0)  # the blue of the night
    shade = np.clip((X - 880 + Y * 0.05) / 170, 0, 1) * trunk_m
    glaze(cv, shade, (0.62, 0.58, 0.64), a=1.0, edge=0.2, gran=0.5, bloom=0.3, seed=212, wob=0)
    limb_shade = np.maximum(limb_m * (1 - trunk_m), front_over)
    glaze(cv, limb_shade, (0.76, 0.74, 0.80), a=0.8, edge=0.2, gran=0.5, bloom=0.3, seed=2121, wob=0)
    glaze(cv, tree_up_m * (noise(16, 213) > 0.15), (0.78, 0.72, 0.68), a=0.8, edge=0.3, gran=0.6, bloom=0, seed=214, wob=0)
    lit = np.clip((900 - X - Y * 0.04) / 60, 0, 1) * trunk_m * (1 - front_over)
    gouache(cv, lit * (noise(6, 2141) > 0.2), (0.78, 0.80, 0.88), op=0.25)  # moonlight on the left edge of the trunk
    moss_root = np.maximum(rootL_m, rootD_m * (X < 1400)) * above_cut * (noise(12, 215) > -0.2)
    moss_root = np.maximum(moss_root, trunk_m * (Y > 540) * (noise(10, 216) > 0.2))
    glaze(cv, moss_root, (0.64, 0.76, 0.40), a=0.9, edge=0.5, edge_w=2, gran=0.6, bloom=0.3, seed=217, wob=1.5, wob_scale=6)
    glaze(cv, light_mouth * rootD_m * above_cut, (1.0, 0.84, 0.56), a=0.9, edge=0, gran=0.2, bloom=0, seed=218, wob=0)

    snow_on = np.zeros((H, W), np.float32)
    for (c, w, w_lin, front) in limb_paths:
        d = np.gradient(c, axis=0)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        nr = np.stack([-d[:, 1], d[:, 0]], 1)
        nr[nr[:, 1] > 0] *= -1
        top_edge = c + nr * w[:, None] / 2
        sw = np.maximum(w * 0.18, 3.0)
        if front:
            # the snow on the front branch continues onto the trunk and tapers to a point
            dep = np.array([trunk_in[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))] for p in top_edge])
            sw = sw * np.clip((30 - dep) / 20, 0, 1) ** 0.6
            band = poly_mask(thick_path_poly_w(top_edge - nr * w[:, None] * 0.06, sw))
        else:
            band = poly_mask(thick_path_poly_w(top_edge - nr * w[:, None] * 0.06, sw)) * (1 - trunk_m)
        snow_on = np.maximum(snow_on, band)
    snow_on = np.maximum(snow_on, poly_mask(thick_path_poly(rootD_c + np.array([0, -19]), 16, 9)))
    snow_on = np.maximum(snow_on, poly_mask(thick_path_poly(rootL_c + np.array([0, -15]), 14, 7)))
    snow_on = wobble(snow_on, 3, 8, 219) * (noise(14, 2190) > -0.45)
    gouache(cv, cv2.GaussianBlur(np.clip(snow_on, 0, 1), (0, 0), 0.8), (0.95, 0.96, 0.99), op=0.9)
    glaze(cv, snow_on * light_mouth, (1.0, 0.86, 0.58), a=0.9, edge=0, gran=0.1, bloom=0, seed=220, wob=0)

    lv = np.zeros((H, W), np.float32)
    rlv = np.random.default_rng(61)
    for (pts, w) in twigs:
        if w > 6 or rlv.random() < 0.55:
            continue
        p = pts[-1]
        for k in range(rlv.integers(1, 4)):
            q = p + rlv.normal(0, 7, 2)
            lv = np.maximum(lv, ellipse_mask(q[0], q[1] + 6, 6, 3.5, rlv.uniform(40, 140)))
    ak.set_reach(F_SKY)  # the leaves end where the sky does
    gouache(cv, lv * sky_m, (0.60, 0.40, 0.25), op=0.85)
    glaze(cv, lv * sky_m * (noise(4, 222) > 0), (0.86, 0.70, 0.56), a=0.6, edge=0, gran=0.3, bloom=0, seed=221, wob=0)

    # ---------- animals
    ak.set_reach(None)
    ink = Ink()
    pen = Pencil()
    paint_animals(cv, light_den / (light_den.max() + 1e-6), ink, pen)

    # ---------- Saturn and the lighthouse
    sat = (2030, 290)
    glow = np.exp(-np.hypot(X - sat[0], Y - sat[1]) / 26) * sky_m
    screen(cv, glow, (0.55, 0.48, 0.30), op=0.6)
    sat_m = ellipse_mask(sat[0], sat[1], 15, 15, 0)
    gouache(cv, sat_m, (0.94, 0.82, 0.58), op=1.0)
    glaze(cv, sat_m * (Y > sat[1] + 2), (0.86, 0.72, 0.56), a=0.8, edge=0, gran=0, bloom=0, seed=251, wob=0)
    lh = (2120, 449)
    lh_m = poly_mask([(lh[0] - 3, lh[1] + 8), (lh[0] - 2, lh[1] - 10), (lh[0] + 2, lh[1] - 10), (lh[0] + 3, lh[1] + 8)])
    glaze(cv, lh_m, (0.42, 0.44, 0.50), a=1.0, edge=0, gran=0, bloom=0, seed=252, wob=0)
    screen(cv, np.exp(-np.hypot(X - lh[0], Y - lh[1] + 10) / 9), (1.0, 0.85, 0.50), op=0.9)
    ak.set_reach(F_SKY)
    for (x, y, rr) in stars:
        if rr >= 2.5 and sky_m[int(y), int(x)] > 0 and keep[int(y), int(x)] > 0.5:
            for dx, dy in [(1, 0), (0, 1)]:
                sm2 = np.zeros((H, W), np.float32)
                cv2.line(sm2, (int((x - dx * 8) * 4), int((y - dy * 8) * 4)), (int((x + dx * 8) * 4), int((y + dy * 8) * 4)), 1.0, 2, cv2.LINE_AA, 2)
                screen(cv, cv2.GaussianBlur(sm2, (0, 0), 0.8), (0.9, 0.9, 0.95), op=0.7)

    sp = np.zeros((H, W), np.float32)
    rsp = np.random.default_rng(808)
    for i in range(420):
        x, y = rsp.uniform(120, W - 120), rsp.uniform(HOR + 60, 640)
        if snowfield_m[int(y), int(x)] > 0.5 or snow_on[int(y), int(x)] > 0.5:
            cv2.circle(sp, (int(x * 4), int(y * 4)), int(rsp.uniform(2, 5)), 1.0, -1, cv2.LINE_AA, 2)
    ak.set_reach(F_SNOW)
    gouache(cv, sp, (1.0, 1.0, 1.0), op=0.8)

    # splatters outside the ends of the washes
    ak.set_reach(None)
    splatter(cv, F_SKY, (Y < HOR - 30).astype(np.float32), (0.52, 0.58, 0.82), 16, 1201)
    splatter(cv, F_EARTH, (Y > gline[None, :] + 70).astype(np.float32), (0.72, 0.58, 0.44), 12, 1202)
    splatter(cv, F_SEA, ((Y > HOR + 8) & (Y < shore[None, :] - 8)).astype(np.float32), (0.72, 0.77, 0.88), 4, 1203)

    # ---------- ink
    xx = np.arange(0, W + 1, 6)
    cut_line = np.stack([xx, g(xx)], 1)
    inside_t = np.array([tun_m[int(min(H - 1, y + 2)), int(min(W - 1, x))] > 0.5 for x, y in cut_line])
    segs, cur = [], []
    for p, ins in zip(cut_line, inside_t):
        if ins:
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        segs.append(np.array(cur))
    for sgm in segs:
        ink.line(sgm, w0=2.4, jitter=1.4, taper=0.02, breaks=0.12)
    # lower edge of the snow layer: broken, thin
    xb = np.arange(0, W + 1, 6)
    sb = np.stack([xb, g(xb) + 30 + 3 * np.sin(xb / 37.0)], 1)
    sb = sb[[hollow_m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] < 0.5 and
             stub_m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] < 0.5 for p in sb]]
    jumps = np.where(np.diff(sb[:, 0]) > 7)[0]
    for prt in np.split(sb, jumps + 1):
        if len(prt) > 4:
            ink.line(prt, w0=1.0, jitter=1.0, taper=0.1, breaks=0.5)
    hb2 = (hollow_m > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(hb2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        pts = cnt[:, 0, :].astype(float)
        keep_pts = pts[pts[:, 1] > g(pts[:, 0]) + 3]
        if len(keep_pts) > 10:
            jumps = np.where(np.linalg.norm(np.diff(keep_pts, axis=0), axis=1) > 5)[0]
            for prt in np.split(keep_pts, jumps + 1):
                if len(prt) > 8:
                    ink.line(prt[::3], w0=2.8, jitter=1.6, taper=0.05, breaks=0.2)
    # the branch outlines start at the trunk edge; the lines of the front branch continue a little onto the trunk
    for (c, w, w_lin, front) in limb_paths:
        d = np.gradient(c, axis=0)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        nr = np.stack([-d[:, 1], d[:, 0]], 1)
        for sgn, wa in ((1, 2.0), (-1, 2.4)):
            pts = c + sgn * nr * w[:, None] / 2
            depth = np.array([trunk_in[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))] for p in pts])
            idx = np.where(depth < (26 if front else 0.5))[0]
            if len(idx) == 0:
                continue
            jumps = np.where(np.diff(idx) > 1)[0]
            for part in np.split(idx, jumps + 1):
                if len(part) > 3:
                    ink.line(pts[part], w0=wa, w1=1.0, jitter=1.0, taper=0.1, breaks=0.2)
    for (pts, w) in twigs:
        ink.line(pts, w0=max(0.7, min(2.6, w * 0.55)), w1=0.5, jitter=0.6, taper=0.35, breaks=0.03)
    # the trunk outline breaks where a branch leaves it
    limb_dil = cv2.dilate((limb_m > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
    tb = (trunk_m > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(tb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        pts = cnt[:, 0, :].astype(float)
        pts = pts[(pts[:, 1] > 2) & (pts[:, 1] < g(pts[:, 0]) - 2)]
        pts = pts[limb_dil[pts[:, 1].astype(int), pts[:, 0].astype(int)] == 0]
        jumps = np.where(np.linalg.norm(np.diff(pts, axis=0), axis=1) > 5)[0]
        for prt in np.split(pts, jumps + 1):
            if len(prt) > 8:
                ink.line(prt[::3], w0=2.8, jitter=1.6, taper=0.05, breaks=0.2)
    rb = np.random.default_rng(33)
    for i in range(150):
        x0 = rb.uniform(800, 1100)
        y0 = rb.uniform(0, 580)
        if trunk_m[int(y0), int(x0)] < 0.5:
            continue
        ln = rb.uniform(30, 120)
        pts = np.array([[x0 + np.sin(k / 6.0 + i) * 2.5, y0 + k * 3] for k in range(int(ln / 3))])
        okb = np.array([trunk_m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] > 0.5 and
                        front_over[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] < 0.3 for p in pts])
        wb = rb.uniform(1.0, 1.8)
        idx = np.where(okb)[0]
        if len(idx) == 0:
            continue
        for part in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(part) > 4:
                ink.line(pts[part], w0=wb, jitter=0.8, taper=0.4, breaks=0.3)
    for (c, w0, w1) in root_paths:
        d = np.gradient(c, axis=0)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        nr = np.stack([-d[:, 1], d[:, 0]], 1)
        t = np.linspace(0, 1, len(c))[:, None]
        w = (w0 * (1 - t) + w1 * t) / 2
        for sgn in (1, -1):
            pts = c + sgn * nr * w
            ok = np.array([(p[1] > g(p[0]) + 32) and hollow_m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] < 0.5 for p in pts])
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            jumps = np.where(np.diff(idx) > 1)[0]
            for part in np.split(idx, jumps + 1):
                if len(part) > 3:
                    ink.line(pts[part], w0=1.7, jitter=0.8, taper=0.2, breaks=0.2)
        # arc of the round tip
        dd = c[-1] - c[-2]
        a0 = np.arctan2(dd[1], dd[0])
        arc = np.array([c[-1] + w1 / 2 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(a0 - np.pi / 2, a0 + np.pi / 2, 9)])
        if all((p[1] > g(p[0]) + 32) and hollow_m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] < 0.5 for p in arc):
            ink.line(arc, w0=1.6, jitter=0.3, taper=0.0, breaks=0.0)
    # the root stubs through the snow layer
    for seg, w0s in stubs:
        dd = seg[-1] - seg[0]
        dd /= np.linalg.norm(dd)
        nr1 = np.array([-dd[1], dd[0]])
        for sgn in (1, -1):
            pts = np.linspace(seg[0] + sgn * nr1 * w0s / 2, seg[-1] + sgn * nr1 * w0s / 2, 40)
            pts = pts[[(g(p[0]) + 2 < p[1] < g(p[0]) + 36) for p in pts]]
            if len(pts) > 2:
                ink.line(pts, w0=1.7, jitter=0.5, taper=0.0, breaks=0.0)
    for p in side_roots:
        pp = p[[hollow_m[int(min(H - 1, q[1])), int(min(W - 1, q[0]))] < 0.5 and q[1] > g(q[0]) + 32 for q in p]]
        if len(pp) > 2:
            ink.line(pp, w0=1.0, w1=0.5, jitter=0.4, taper=0.5, breaks=0.0)
    d = np.gradient(rootD_c, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nr = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(rootD_c))[:, None]
    w = (40 * (1 - t) + 14 * t) / 2
    ink.line((rootD_c - nr * w)[2:], w0=2.2, jitter=1.0, taper=0.1, breaks=0.15)
    ink.line((rootD_c + nr * w)[2:], w0=1.6, jitter=1.0, taper=0.1, breaks=0.3)
    rh = np.random.default_rng(91)
    for i in range(90):
        x0 = rh.uniform(700, 1480)
        cy = None
        for yy in range(700, 960):
            if den_m[yy, int(x0)] > 0.5:
                cy = yy
                break
        if cy is None:
            continue
        ln = rh.uniform(10, 50)
        pts = [(x0, cy - 2)]
        a = np.pi / 2 + rh.normal(0, 0.25)
        for k in range(int(ln / 4)):
            a += rh.normal(0, 0.25)
            pts.append((pts[-1][0] + 4 * np.cos(a), pts[-1][1] + 4 * np.sin(a)))
        ink.line(pts, w0=rh.uniform(0.6, 1.2), w1=0.4, jitter=0.3, taper=0.5)
    for (x, y, rr) in CUTS:
        cc = [(x + rr * np.cos(a), y + rr * 0.85 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 30)]
        ink.line(cc, w0=1.6, jitter=0.4, taper=0.0)
        cc2 = [(x + rr * 0.5 * np.cos(a), y + rr * 0.42 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20)]
        ink.line(cc2, w0=0.8, jitter=0.3, taper=0.0)
    for (x, y, rx, ry, a) in STONES:
        aa = np.deg2rad(a)
        cc = [(x + rx * np.cos(t) * np.cos(aa) - ry * np.sin(t) * np.sin(aa), y + rx * np.cos(t) * np.sin(aa) + ry * np.sin(t) * np.cos(aa))
              for t in np.linspace(0.3, 2 * np.pi * 0.85, 18)]
        ink.line(cc, w0=1.2, jitter=0.6, taper=0.2)
    ink.line(catmull(BOULDER, closed=True, n=6), w0=2.0, jitter=1.2, taper=0.02, breaks=0.15)
    ink.line([(1790, 1030), (1812, 1064), (1806, 1096), (1830, 1124)], w0=1.1, jitter=0.6, taper=0.3)
    ink.line([(1900, 1040), (1920, 1070), (1958, 1086)], w0=1.0, jitter=0.6, taper=0.3)
    ink.line(catmull(BOULD, closed=False, n=10), w0=2.0, jitter=1.2, taper=0.1, breaks=0.2)
    lx, ly = LANTERN
    ink.line([(lx, 800), (lx, 838)], w0=1.3, jitter=0.2, taper=0.0)
    ink.line([(lx - 20, 846), (lx + 20, 846), (lx + 16, 906), (lx - 16, 906), (lx - 20, 846)], w0=2.4, jitter=0.3, taper=0.0)
    ink.line([(lx - 13, 840), (lx, 828), (lx + 13, 840)], w0=2.0, jitter=0.2, taper=0.0)
    ink.line([(lx - 22, 908), (lx + 22, 908)], w0=2.6, jitter=0.2, taper=0.0)
    ink.line([(lx, 846), (lx, 906)], w0=1.0, jitter=0.2, taper=0.0)
    for (bx, by, bw, bh, bc) in books:
        ink.line([(bx, by + bh / 2), (bx, by - bh / 2), (bx + bw, by - bh / 2), (bx + bw, by + bh / 2)], w0=1.2, jitter=0.3, taper=0.0)
    ink.line(catmull(niche, closed=True, n=8), w0=1.6, jitter=0.6, taper=0.05)
    sx, sy = sat
    ink.line([(sx + 15 * np.cos(t), sy + 15 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 40)], w0=1.2, jitter=0.2, taper=0.0)
    ring = [(sx + 34 * np.cos(t) * np.cos(-0.35) - 9 * np.sin(t) * np.sin(-0.35),
             sy + 34 * np.cos(t) * np.sin(-0.35) + 9 * np.sin(t) * np.cos(-0.35)) for t in np.linspace(0, 2 * np.pi, 60)]
    ring = [p for p in ring if not (np.hypot(p[0] - sx, p[1] - sy) < 15 and p[1] < sy + 2)]
    ink.line(ring, w0=1.3, jitter=0.2, taper=0.0)

    soil_dens = np.clip((Y - gline[None, :] - 60) / 600, 0, 1) * 0.75 + 0.12
    soil_dens = soil_dens * (1 - cv2.GaussianBlur(hollow_m, (0, 0), 40) * 1.5)
    soil_dens = soil_dens * np.clip(F_EARTH / 70, 0, 1)  # the hatching thins out before the earth color ends
    hatch(ink, soil * hmask * (1 - st_m) * (1 - boulder_m) * (1 - roots_m), angle=np.deg2rad(-35), spacing=10, length=32, w=0.95,
          seed=301, density=np.clip(soil_dens, 0, 1), breaks=0.5)
    hatch(ink, trunk_m * (X > 900 + Y * 0.06), angle=np.deg2rad(84), spacing=7, length=40, w=0.85, seed=302,
          density=np.clip((X - 880) / 150, 0, 1), breaks=0.4)
    hatch(ink, hollow_m * (1 - m['fox_all']) * (1 - m['lumi_full']) * (1 - bed_m) * (1 - niche_m), angle=np.deg2rad(-50), spacing=9,
          length=26, w=0.85, seed=303, density=np.clip(1 - light_den / (light_den.max() + 1e-6) * 2.2, 0, 1) * 0.9, breaks=0.4)
    hatch(ink, boulder_m, angle=np.deg2rad(35), spacing=8, length=24, w=0.8, seed=304, density=np.clip((Y - 1060) / 100, 0, 1), breaks=0.4)

    # ink and colored pencil reach only where color has spread too (small splatters do not count)
    paper = new_canvas()
    painted = (np.max(np.abs(1 - cv / np.maximum(paper, 1e-3)), axis=2) > 0.05).astype(np.uint8)
    painted = cv2.morphologyEx(painted, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(painted, connectivity=8)
    big = np.isin(lab, np.where(stats[:, cv2.CC_STAT_AREA] > 4000)[0]) & (lab > 0)
    reach_ink = cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ak.set_ink_reach(cv2.GaussianBlur(reach_ink.astype(np.float32), (0, 0), 3))
    ink.apply(cv, col=(0.24, 0.15, 0.09), op=0.82)

    # ---------- colored pencil
    rp = np.random.default_rng(402)
    for i in range(1500):
        x = rp.uniform(680, 1500)
        y = rp.uniform(1150, 1215)
        if bed_m[int(min(H - 1, y)), int(x)] < 0.5:
            continue
        pen.stroke('moss', [(x, y), (x + rp.uniform(-3, 3), y - rp.uniform(2, 6))], w=1.0)
    for i in range(260):
        x = rp.choice([rp.uniform(160, 600), rp.uniform(1800, 2300)])
        base = g(x) - rp.uniform(2, 34)
        hgt = rp.uniform(14, 46)
        lean = rp.uniform(-0.5, 0.2)
        pen.stroke('grass', [(x, base), (x + lean * hgt * 0.5, base - hgt * 0.55), (x + lean * hgt, base - hgt)], w=1.0)
    pen.apply(cv, 'fox_dark', (0.62, 0.32, 0.14), op=0.5)
    pen.apply(cv, 'fox_light', (1.0, 0.78, 0.48), op=0.45)
    pen.apply(cv, 'white_fur', (0.99, 0.98, 0.96), op=0.55, mode='over')
    pen.apply(cv, 'lumi_fur', (0.74, 0.76, 0.86), op=0.3)
    pen.apply(cv, 'moss', (0.40, 0.52, 0.22), op=0.6)
    pen.apply(cv, 'grass', (0.80, 0.64, 0.36), op=0.75)

    # ---------- light last
    glass = poly_mask([(lx - 18, 848), (lx + 18, 848), (lx + 14, 904), (lx - 14, 904)])
    gouache(cv, glass, (1.0, 0.88, 0.55), op=0.85)
    gouache(cv, ellipse_mask(lx, ly, 6, 12, 0, blur=1.2), (1.0, 0.97, 0.82), op=1.0)
    screen(cv, np.exp(-np.hypot(X - lx, Y - ly) / 60) * hollow_m, (0.55, 0.42, 0.18), op=0.6)
    ex_, ey_ = hm.LUMI_EYE
    cv2.circle(cv, (int(ex_ + 1.5), int(ey_ - 1.5)), 1, (0.98, 0.98, 0.98), -1, cv2.LINE_AA)

    save(cv, out)


if __name__ == '__main__':
    render(sys.argv[1] if len(sys.argv) > 1 else 'v2.jpg')
