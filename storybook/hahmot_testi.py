"""A quick test of the characters: only the den wall, the fox and Lumi."""
import sys
import numpy as np
import cv2
from akvarelli import (W, H, X, Y, noise, new_canvas, catmull, poly_mask, ellipse_mask, glaze, gouache, screen,
                       Ink, Pencil, save, PAPER_HF)
import hahmot as hm


def paint_animals(cv, light, ink, pen, op_ink=True):
    m = hm.masks()
    fox = m['fox_all']
    # 1. base color: warm orange, lighter on the lit side
    glaze(cv, fox, (1.0, 0.76, 0.48), a=1.0, edge=0.35, edge_w=5, gran=0.12, bloom=0.25, bloom_scale=40, seed=301, wob=0.8, wob_scale=8)
    glaze(cv, fox, (0.97, 0.66, 0.40), a=1.0, edge=0.3, edge_w=5, gran=0.12, bloom=0.3, bloom_scale=30, seed=302, wob=0.8, wob_scale=8,
          light=light, light_k=0.55)
    # 2. form: shadow farther from the light and under the tail
    shade = np.clip(1 - light * 1.8, 0, 1)
    glaze(cv, fox * shade, (0.84, 0.56, 0.40), a=0.9, edge=0.1, gran=0.12, bloom=0.25, seed=303, wob=0)
    # the darker stripe along the back
    back_line = cv2.GaussianBlur(poly_mask(np.vstack([catmull(hm.BODY, closed=True, n=12)[:int(17 * 12 * 0.55)]])), (0, 0), 1)
    ridge = np.zeros((H, W), np.float32)
    b = catmull(hm.BODY, closed=True, n=12)[: int(len(catmull(hm.BODY, closed=True, n=12)) * 0.55)]
    for i in range(len(b) - 1):
        cv2.line(ridge, tuple((b[i] * 4).astype(int)), tuple((b[i + 1] * 4).astype(int)), 1.0, 4 * 26, cv2.LINE_AA, 2)
    ridge = cv2.GaussianBlur(ridge, (0, 0), 10) * m['body']
    glaze(cv, ridge, (0.82, 0.50, 0.34), a=0.8, edge=0, gran=0.1, bloom=0.2, seed=304, wob=0)
    # shadow of the tail's edge on the back and on Lumi (contact shadow)
    tail_sh = cv2.GaussianBlur(m['tail_full'], (0, 0), 9) * (1 - m['tail_full'])
    glaze(cv, tail_sh * (m['body'] + m['lumi']), (0.80, 0.62, 0.56), a=0.9, edge=0, gran=0.1, bloom=0, seed=305, wob=0)
    # the white parts
    gouache(cv, cv2.GaussianBlur(np.maximum(m['tail_tip'], m['cheek']), (0, 0), 1.2), (0.975, 0.96, 0.92), op=0.95)
    glaze(cv, np.maximum(m['tail_tip'], m['cheek']) * shade, (0.86, 0.87, 0.95), a=0.7, edge=0.2, gran=0.1, bloom=0, seed=306, wob=0)
    # backs of the ears and the paw are dark
    ear_dark = m['ears'] * (Y < hm.CY - 82 * hm.SC)
    glaze(cv, m['ears'] * (Y >= hm.CY - 82 * hm.SC), (0.96, 0.80, 0.62), a=0.6, edge=0, gran=0.1, bloom=0, seed=3071, wob=0)
    glaze(cv, ear_dark, (0.42, 0.32, 0.28), a=1.0, edge=0.3, gran=0.15, bloom=0, seed=307, wob=0.4, wob_scale=3)
    glaze(cv, m['paw'], (0.52, 0.40, 0.36), a=0.9, edge=0.4, gran=0.15, bloom=0, seed=308, wob=0.4, wob_scale=3)

    # Lumi: paper white, the lit side warm, the shadow side bluish, with the fox's orange reflected on the back
    lumi = m['lumi']
    gouache(cv, cv2.GaussianBlur(lumi, (0, 0), 0.9), (0.975, 0.965, 0.935), op=0.97)
    lb = (m['lumi_full'] > 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(lb, cv2.DIST_L2, 3)
    # round form: the edges darken, the right (front) side is cool
    rim = np.exp(-dist_in / 7.0) * lumi
    right_side = np.clip((X - hm.LB[0] - 2) / 24, 0, 1)
    glaze(cv, rim * right_side, (0.80, 0.82, 0.93), a=0.9, edge=0, gran=0.05, bloom=0, seed=311, wob=0)
    glaze(cv, lumi * right_side * 0.6, (0.90, 0.91, 0.97), a=0.8, edge=0, gran=0.05, bloom=0, seed=312, wob=0)
    left_side = np.clip((hm.LB[0] + 4 - X) / 20, 0, 1)
    glaze(cv, rim * left_side, (1.0, 0.82, 0.62), a=0.85, edge=0, gran=0.05, bloom=0, seed=313, wob=0)
    neck_sh = lumi * cv2.GaussianBlur(ellipse_mask(hm.LB[0] + 26, hm.LB[1] - 96, 14, 8, -30), (0, 0), 5)
    glaze(cv, neck_sh, (0.84, 0.84, 0.93), a=0.8, edge=0, gran=0.05, bloom=0, seed=314, wob=0)
    nose_pink = ellipse_mask(hm.LUMI_NOSE[0] - 1, hm.LUMI_NOSE[1] + 1, 3, 2.2, -38)
    glaze(cv, nose_pink, (0.95, 0.70, 0.70), a=0.9, edge=0, gran=0, bloom=0, seed=315, wob=0)

    # ink: outlines are cut where something is in front of them
    def clip_runs(pts, mask):
        pts = np.asarray(pts, float)
        ok = np.array([mask[int(min(H - 1, max(0, p[1]))), int(min(W - 1, max(0, p[0])))] < 0.5 for p in pts])
        runs, cur = [], []
        for p, k in zip(pts, ok):
            if k:
                cur.append(p)
            else:
                if len(cur) > 2:
                    runs.append(np.array(cur))
                cur = []
        if len(cur) > 2:
            runs.append(np.array(cur))
        return runs
    head_all = np.maximum(m['head'], m['ears'])
    front_of_body = np.clip(m['tail_full'] + m['lumi_full'] + head_all, 0, 1)
    paths = hm.ink_paths()
    for k, (pts, w) in enumerate(paths):
        if k in (0, 1, 2):        # back, thigh, shoulder: the tail, Lumi and the head are in front
            cm = cv2.dilate(front_of_body, np.ones((3, 3), np.uint8))
        elif k in (3, 4):         # edges of the tail: the head is in front
            cm = cv2.dilate(head_all, np.ones((3, 3), np.uint8))
        else:
            cm = np.zeros_like(head_all)
        for run in clip_runs(pts, cm):
            ink.line(run, w0=w, jitter=0.7, taper=0.06, breaks=0.05)
    lb = catmull(hm.LUMI_OUTLINE, closed=True, n=10)
    tm = m['tail_full']
    vis = [p for p in lb if tm[int(p[1]), int(p[0])] < 0.5]
    ink.line(vis, w0=1.5, jitter=0.4, taper=0.08, breaks=0.03)
    ex, ey, er = hm.LUMI_EAR
    ink.line([(ex + er * np.cos(t), ey + er * 0.9 * np.sin(t)) for t in np.linspace(2.4, 6.3, 20)], w0=1.2, jitter=0.2, taper=0.1)
    ink.dot(*hm.LUMI_EYE, 4.6)
    ink.dot(*hm.LUMI_NOSE, 2.0)
    # whiskers
    for k in range(3):
        a0 = -0.15 + k * 0.22
        x0, y0 = hm.LUMI_NOSE[0] - 8, hm.LUMI_NOSE[1] + 6
        ink.line([(x0, y0), (x0 + 14 * np.cos(a0), y0 + 14 * np.sin(a0) + 2)], w0=0.5, jitter=0.1, taper=0.4)
    ink.dot(*hm.NOSE, 5.0)
    for e in hm.LUMI_PAWS:
        cx, cy, rx, ry, a = e
        aa = np.deg2rad(a)
        ink.line([(cx + rx * np.cos(t) * np.cos(aa) - ry * np.sin(t) * np.sin(aa), cy + rx * np.cos(t) * np.sin(aa) + ry * np.sin(t) * np.cos(aa))
                  for t in np.linspace(0, 2 * np.pi, 16)], w0=1.0, jitter=0.2, taper=0.0)

    # colored pencil: fur
    r = np.random.default_rng(401)
    tail_cl = hm.TAIL_CL
    for i in range(1300):
        x = r.uniform(hm.CX - 260, hm.CX + 360)
        y = r.uniform(hm.CY - 160, hm.CY + 190)
        if m['fox_all'][int(y), int(x)] < 0.6 or m['lumi'][int(y), int(x)] > 0.3:
            continue
        if m['tail_full'][int(y), int(x)] > 0.5 and m['head'][int(y), int(x)] < 0.5:
            j = np.argmin(np.hypot(tail_cl[:, 0] - x, tail_cl[:, 1] - y))
            dd = tail_cl[min(j + 1, len(tail_cl) - 1)] - tail_cl[max(j - 1, 0)]
            ang = np.arctan2(dd[1], dd[0]) + r.normal(0, 0.25)
        elif m['head'][int(y), int(x)] > 0.5:
            ang = np.arctan2(y - (hm.CY - 10), x - (hm.CX + 270)) + r.normal(0, 0.3)
        else:
            ang = np.arctan2(y - hm.CY, x - hm.CX) + np.pi / 2 + r.normal(0, 0.2)
        ln = r.uniform(7, 16)
        pts = [(x, y), (x + np.cos(ang) * ln * 0.5, y + np.sin(ang) * ln * 0.5 + 0.8), (x + np.cos(ang) * ln, y + np.sin(ang) * ln + 1.6)]
        if m['tail_tip'][int(y), int(x)] > 0.5 or m['cheek'][int(y), int(x)] > 0.5:
            key = 'white_fur'
        else:
            key = 'fox_dark' if r.random() < 0.5 else 'fox_light'
        pen.stroke(key, pts, w=0.9)
    for i in range(220):
        x = r.uniform(hm.CX - 40, hm.CX + 90)
        y = r.uniform(hm.CY - 150, hm.CY + 90)
        if m['lumi'][int(y), int(x)] < 0.6:
            continue
        ln = r.uniform(4, 8)
        ang = -1.2 + r.normal(0, 0.2)
        pen.stroke('lumi_fur', [(x, y), (x + np.cos(ang) * ln, y + np.sin(ang) * ln)], w=0.7)
    return m


if __name__ == '__main__':
    cv = new_canvas()
    lantern = (800, 835)
    dl = np.hypot(X - lantern[0], Y - lantern[1])
    light = np.exp(-dl / 340.0)
    bg = poly_mask([(620, 760), (1500, 760), (1500, 1250), (620, 1250)])
    glaze(cv, bg, (0.98, 0.86, 0.62), a=1.0, edge=0.5, gran=0.3, seed=1, wob=0)
    glaze(cv, bg * np.clip(1 - light * 1.4, 0, 1), (0.80, 0.62, 0.42), a=1.0, edge=0, gran=0.4, seed=2, wob=0)
    ink = Ink()
    pen = Pencil()
    paint_animals(cv, light, ink, pen)
    ink.apply(cv, col=(0.24, 0.15, 0.09), op=0.85)
    pen.apply(cv, 'fox_dark', (0.62, 0.32, 0.14), op=0.5)
    pen.apply(cv, 'fox_light', (1.0, 0.78, 0.48), op=0.45)
    pen.apply(cv, 'white_fur', (0.99, 0.98, 0.96), op=0.55, mode='over')
    pen.apply(cv, 'lumi_fur', (0.74, 0.76, 0.86), op=0.3)
    cv2.circle(cv, (int(hm.LUMI_EYE[0] + 1.5), int(hm.LUMI_EYE[1] - 1.5)), 1, (0.98, 0.98, 0.98), -1, cv2.LINE_AA)
    out = sys.argv[1] if len(sys.argv) > 1 else 'hahmot.jpg'
    save(cv[820:1220, 780:1440], out)
