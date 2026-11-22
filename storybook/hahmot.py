"""Kettu (the fox) and Lumi (the weasel): shapes, masks and ink paths.

The fox sleeps curled up: the body is a round mound, the tail wraps around
the front, and the head rests on the tail on the right. Lumi sits in the
fox's lap behind the tail and looks up and to the right, towards the tunnel.
"""
import numpy as np
import cv2
from akvarelli import catmull, poly_mask, ellipse_mask, X, Y, noise

CX, CY = 1066, 1004  # center point of the fox
SC = 1.1  # size


def P(pts):
    return [(CX + x * SC, CY + y * SC) for x, y in pts]


def S(pts, c, k):
    return [(c[0] + (x - c[0]) * k, c[1] + (y - c[1]) * k) for x, y in pts]


BODY = P([(-190, 86), (-206, 30), (-200, -30), (-164, -82), (-94, -120), (0, -134), (82, -128), (152, -106),
          (206, -72), (238, -34), (248, 8), (240, 86)])
THIGH = P([(-198, -36), (-184, 2), (-154, 30)])
SHOULDER = P([(166, -90), (192, -56), (202, -22)])
TAIL_CL = catmull(P([(-150, 18), (-188, 52), (-160, 98), (-60, 122), (40, 120), (130, 104), (196, 76), (232, 44)]),
                  closed=False, n=18)
HC = (CX + 246 * SC, CY - 2 * SC)
HK = 1.32
HEAD = S(P([(182, 12), (188, -4), (202, -22), (226, -40), (256, -48), (286, -40), (305, -18), (308, 12), (294, 33),
            (262, 43), (226, 41), (198, 29)]), HC, HK)
EAR_FAR = S(P([(230, -40), (240, -92), (258, -46)]), HC, HK)
EAR_NEAR = S(P([(266, -44), (290, -88), (296, -30)]), HC, HK)
PAW = S(P([(226, 34), (256, 37), (264, 45), (256, 50), (230, 48)]), HC, HK)
EYE = S(P([(208, -10), (214, -5), (222, -4), (230, -7)]), HC, HK)
NOSE = S(P([(182, 8)]), HC, HK)[0]

# Lumi: a stoat-like silhouette, muzzle pointing up and to the right
LB = (CX + 6 * SC, CY + 66 * SC)
LK = 1.1 * SC


def L(pts):
    return [(LB[0] + x * LK, LB[1] + y * LK) for x, y in pts]


LUMI_OUTLINE = L([(-22, 30), (-27, -4), (-26, -36), (-20, -60), (-12, -78), (-12, -94), (-6, -108), (4, -118),
                  (18, -124), (30, -124), (40, -126), (48, -130), (54, -133), (55, -128), (50, -122), (40, -117),
                  (32, -110), (26, -102), (22, -94), (25, -78), (29, -56), (27, -26), (24, 30)])
LUMI_EAR = (LB[0] + 2 * LK, LB[1] - 116 * LK, 9.0 * LK)
LUMI_EYE = (LB[0] + 30 * LK, LB[1] - 113 * LK)
LUMI_NOSE = (LB[0] + 54 * LK, LB[1] - 131 * LK)
LUMI_PAWS = [(LB[0] + 27 * LK, LB[1] - 60 * LK, 7.5, 4.2, -75), (LB[0] + 20 * LK, LB[1] - 56 * LK, 7.5, 4.2, -75)]


def tail_poly(c=TAIL_CL, seed=7):
    n = len(c)
    t = np.linspace(0, 1, n)
    wid = (40 + 62 * np.sin(np.pi * np.clip(t * 1.05, 0, 1)) - 6 * t) * SC
    # fluffy edge: small tufts
    r = np.random.default_rng(seed)
    wl = wid * (1 + 0.06 * np.sin(t * 38 + 1.3) + 0.03 * r.standard_normal(n))
    wr = wid * (1 + 0.06 * np.sin(t * 41 + 0.4) + 0.03 * r.standard_normal(n))
    d = np.gradient(c, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nr = np.stack([-d[:, 1], d[:, 0]], 1)
    left = c + nr * wl[:, None] / 2
    right = c - nr * wr[:, None] / 2
    tip = c[-1] + d[-1] * 30 * SC
    return np.vstack([left, [tip], right[::-1]]), wid


def masks():
    body = poly_mask(catmull(BODY, closed=True, n=12))
    tp, _ = tail_poly()
    tail = poly_mask(catmull(tp, closed=True, n=3))
    head = poly_mask(catmull(HEAD, closed=True, n=12))
    ears = np.maximum(poly_mask(EAR_FAR), poly_mask(EAR_NEAR))
    paw = poly_mask(catmull(PAW, closed=True, n=8)) * 0
    n = len(TAIL_CL)
    k = int(n * 0.80)
    tip_start = TAIL_CL[k]
    d = TAIL_CL[min(k + 2, n - 1)] - TAIL_CL[k]
    proj = (X - tip_start[0]) * d[0] + (Y - tip_start[1]) * d[1]
    tail_tip = tail * np.clip(proj / 40 + 0.5, 0, 1) * (1 + 0.5 * noise(6, 12)).clip(0.6, 1.4)
    tail_tip = np.clip(cv2.GaussianBlur(tail_tip, (0, 0), 4), 0, 1) * tail
    lb = poly_mask(catmull(LUMI_OUTLINE, closed=True, n=10))
    le = ellipse_mask(LUMI_EAR[0], LUMI_EAR[1], LUMI_EAR[2], LUMI_EAR[2] * 0.9, 0)
    lumi = np.maximum(lb, le)
    lp = np.zeros_like(lumi)
    for e in LUMI_PAWS:
        lp = np.maximum(lp, ellipse_mask(*e))
    head_all = np.maximum(head, ears)
    tail_vis = tail * (1 - head_all) * (1 - paw)
    lumi_vis = lumi * (1 - tail) * (1 - head_all)
    body_vis = body * (1 - tail) * (1 - lumi) * (1 - head_all)
    cheek_pts = S(P([(180, 6), (196, 14), (222, 21), (252, 24), (280, 20), (300, 12), (312, 14), (312, 40), (292, 50),
                     (258, 52), (222, 50), (194, 36)]), HC, HK)
    cheek = head * poly_mask(catmull(cheek_pts, closed=True, n=10))
    cheek = cv2.GaussianBlur(cheek, (0, 0), 1.5) * head
    return dict(body=body_vis, tail=tail_vis, tail_full=tail, tail_tip=tail_tip * (1 - head_all), head=head, ears=ears,
                paw=paw, cheek=cheek, lumi=lumi_vis, lumi_full=lumi, lumi_paws=lp * (1 - tail),
                fox_all=np.maximum.reduce([body, tail, head_all, paw]))


def ink_paths():
    out = []
    b = catmull(BODY, closed=True, n=12)
    out.append((b[: int(len(b) * 0.80)], 2.4))
    out.append((catmull(THIGH, closed=False, n=10), 1.5))
    out.append((catmull(SHOULDER, closed=False, n=10), 1.3))
    tp, _ = tail_poly()
    n = len(TAIL_CL)
    out.append((tp[: n + 1], 2.2))
    out.append((tp[n:], 2.2))
    out.append((catmull(HEAD, closed=True, n=12), 2.2))
    out.append((EAR_FAR, 1.9))
    out.append((EAR_NEAR + [EAR_NEAR[0]], 1.9))
    out.append((EYE, 2.0))
    return out
