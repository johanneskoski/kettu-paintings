"""Watercolor, ink and colored pencil on paper.

A small painting machine for a storybook picture. Everything happens in
layers: glazes (transparent color is multiplied onto the paper), opaque
gouache, an ink pen and a colored pencil. The light is planned: lit areas
get less glaze, so the paper stays visible.
"""
import numpy as np
import cv2

W, H = 2400, 1350
SS = 3  # supersampling for the ink

_X, _Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
X, Y = _X, _Y


def noise(scale, seed, octaves=1):
    """Value noise in the range of about -1..1, with features about `scale` pixels in size."""
    r = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(round(scale / (2 ** o))))
        gh, gw = H // s + 4, W // s + 4
        g = r.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (gw * s, gh * s), interpolation=cv2.INTER_CUBIC)[:H, :W]
        out += amp * up
        tot += amp
        amp *= 0.5
    out /= tot
    sd = out.std() + 1e-6
    return np.clip(out / (2.2 * sd), -1, 1)


# The paper surface: high spots and pits. Pigment settles into the pits
# (granulation), and the colored pencil catches only on the high spots.
PAPER_HF = np.clip(0.65 * noise(2.2, 101) + 0.35 * noise(6, 102), -1, 1)
PAPER_MF = noise(30, 103, octaves=3)
PAPER = np.array([0.965, 0.935, 0.86], np.float32)


def new_canvas():
    cv = np.ones((H, W, 3), np.float32) * PAPER[None, None, :]
    # the paper's own fibre and a slight yellowing
    cv *= (1 - 0.018 * (PAPER_HF[..., None] + 1) * 0.5)
    cv *= (1 - 0.012 * PAPER_MF[..., None])
    return cv


def catmull(points, closed=True, n=14):
    """A smooth curve through the control points."""
    p = np.asarray(points, np.float64)
    if closed:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(p[-2])
    return np.array(out)


def poly_mask(pts, blur=0.0):
    m = np.zeros((H, W), np.uint8)
    p = np.round(np.asarray(pts) * 16).astype(np.int32)
    cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_AA, shift=4)
    m = m.astype(np.float32) / 255.0
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m


def ellipse_mask(cx, cy, rx, ry, ang=0.0, blur=0.0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx * 16), int(cy * 16)), (int(rx * 16), int(ry * 16)), ang, 0, 360, 255, -1, cv2.LINE_AA, 4)
    m = m.astype(np.float32) / 255.0
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m


def thick_path_poly(center, w0, w1=None):
    """A polygon made from a thick, tapering line (roots, branches, the tunnel)."""
    c = np.asarray(center, np.float64)
    if w1 is None:
        w1 = w0
    d = np.gradient(c, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(c))[:, None]
    w = (w0 * (1 - t) + w1 * t) / 2
    left = c + nrm * w
    right = c - nrm * w
    return np.vstack([left, right[::-1]])


def wobble(m, amp, scale, seed):
    if amp <= 0:
        return m
    nx = noise(scale, seed)
    ny = noise(scale, seed + 1)
    return cv2.remap(m, X + nx * amp, Y + ny * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


VIGNETTE = None


def set_vignette(v):
    global VIGNETTE
    VIGNETTE = v


def _vig(m):
    return m if VIGNETTE is None else m * VIGNETTE


# ---------------------------------------------------------------- wash edges
# The picture does not fade to white at the edges. Each wash ends by itself.
# Every area (sky, sea, earth ...) has its own reach field F in pixels:
# F > 0 where the wash reaches. A wet edge leaves a dark border, a dry brush
# breaks up on the grain of the paper, and in wet-on-wet spots the edge
# softens. Each wash layer ends in a slightly different place.
REACH = None
INK_REACH = None
DRY = None
SOFT = None
STREAK = None
RIM_K = 0.55
RIM_W = 4.5
_count = [0]


def init_edges(seed=500):
    global DRY, SOFT, STREAK
    DRY = np.clip((noise(230, seed, octaves=2) - 0.10) * 2.5, 0, 1)
    SOFT = np.clip((noise(260, seed + 1, octaves=2) - 0.20) * 2.5, 0, 1) * (1 - DRY)
    # horizontal streaks of the dry brush
    r = np.random.default_rng(seed + 2)
    g = r.standard_normal((H // 2 + 4, W // 16 + 4)).astype(np.float32)
    st = cv2.resize(g, ((W // 16 + 4) * 16, (H // 2 + 4) * 2), interpolation=cv2.INTER_CUBIC)[:H, :W]
    STREAK = np.clip(st / (2.2 * st.std()), -1, 1)


def set_reach(F):
    """Sets the reach field of the current area (or None: no limit)."""
    global REACH
    REACH = F
    if F is not None and DRY is None:
        init_edges()


def set_ink_reach(m):
    global INK_REACH
    INK_REACH = m


def _reach(seed=None):
    if REACH is None:
        return None
    if seed is None:
        _count[0] += 1
        seed = 70000 + _count[0]
    rr = np.random.default_rng(seed + 9001)
    dR = REACH - rr.uniform(-4, 14) - 6 * noise(60, seed + 9002)
    rm = np.clip(dR / (1.0 + 9.0 * SOFT) + 0.5, 0, 1)
    near = np.clip(dR / 36.0, 0, 1)
    keep = np.clip((0.5 * PAPER_HF + 0.5 * STREAK + 1.2 * near - 0.3) * 3.5, 0, 1)
    rm = rm * (1 - DRY * (1 - keep))
    rim = np.exp(-np.clip(dR, 0, None) / RIM_W) * (dR > -0.5) * (1 - DRY) * (1 - SOFT)
    return rm, rim


def glaze(cv, m, col, a=0.5, edge=0.6, edge_w=4.0, gran=0.35, bloom=0.25, bloom_scale=70,
          seed=0, wob=3.0, wob_scale=40, light=None, light_k=0.0):
    """A transparent glaze. `col` is the transmission color of the pigment (1 = no color).

    Pigment gathers at the edge (wet edge), pigment settles into the pits of
    the paper, and large washes have slow variation (blooms).
    `light` reduces the pigment in lit areas.
    """
    mm = wobble(m, wob, wob_scale, seed) if wob else m
    rch = _reach(seed)
    if rch is not None:
        mm = mm * rch[0]
    mb = (mm > 0.5).astype(np.uint8)
    mm = _vig(mm)
    d = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    e = np.exp(-d / edge_w) * mb
    dens = a * (1 + edge * e)
    if rch is not None:
        dens = dens * (1 + RIM_K * rch[1])
    if bloom:
        dens = dens * (1 + bloom * noise(bloom_scale, seed + 11, octaves=2))
    if gran:
        dens = dens * (1 + gran * PAPER_HF)
    if light is not None and light_k:
        dens = dens * np.clip(1 - light_k * light, 0, 1)
    dens = np.clip(dens, 0, 1.6) * mm
    col = np.asarray(col, np.float32)
    if col.ndim == 1:
        col = col[None, None, :]
    cv *= (1 - dens[..., None] * (1 - col))
    np.clip(cv, 0, 1, out=cv)
    return cv


def gouache(cv, m, col, op=1.0, seed=None):
    rch = _reach(seed)
    if rch is not None:
        m = m * rch[0]
    m = _vig(m)
    col = np.asarray(col, np.float32)
    if col.ndim == 1:
        col = col[None, None, :]
    a = np.clip(m * op, 0, 1)[..., None]
    cv[:] = cv * (1 - a) + col * a
    return cv


def screen(cv, m, col, op=1.0, seed=None):
    rch = _reach(seed)
    if rch is not None:
        m = m * rch[0]
    m = _vig(m)
    col = np.asarray(col, np.float32)
    if col.ndim == 1:
        col = col[None, None, :]
    a = np.clip(m * op, 0, 1)[..., None]
    scr = 1 - (1 - cv) * (1 - col)
    cv[:] = cv * (1 - a) + scr * a
    return cv


class Ink:
    """A supersampled ink layer. The lines wobble like lines drawn by hand."""

    def __init__(self):
        self.layer = np.zeros((H * SS, W * SS), np.uint8)
        self.rng = np.random.default_rng(5)

    def line(self, pts, w0=2.0, w1=None, jitter=0.8, taper=0.25, breaks=0.0, pressure=0.25, seed=None):
        r = self.rng if seed is None else np.random.default_rng(seed)
        p = np.asarray(pts, np.float64)
        if len(p) < 2:
            return
        # even resampling, about 3 px apart
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        L = seg.sum()
        if L < 1:
            return
        n = max(2, int(L / 3))
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.linspace(0, L, n)
        q = np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)
        # wobble of the hand: a slow sideways drift
        if jitter:
            k = np.cumsum(r.standard_normal((n, 2)), 0)
            k = cv2.GaussianBlur(k.astype(np.float32), (1, 0), 6).astype(np.float64) if n > 12 else k
            k -= np.linspace(k[0], k[-1], n)
            k = k / (np.abs(k).max() + 1e-9) * jitter
            q = q + k
        t = np.linspace(0, 1, n)
        if w1 is None:
            w1 = w0
        w = w0 * (1 - t) + w1 * t
        if taper:
            tp = np.minimum(1, np.minimum(t, 1 - t) / taper) if taper > 0 else 1
            w = w * (0.35 + 0.65 * tp)
        if pressure:
            kk = min(9, n)
            pr = 1 + pressure * np.convolve(r.standard_normal(n), np.ones(kk) / kk, mode='same')[:n] * 2
            w = w * np.clip(pr, 0.4, 1.6)
        alive = np.ones(n, bool)
        if breaks:
            i = 0
            while i < n:
                if r.random() < breaks / 10:
                    gap = r.integers(2, 6)
                    alive[i:i + gap] = False
                    i += gap
                i += 1
        for i in range(n - 1):
            if not (alive[i] and alive[i + 1]):
                continue
            a = (int(q[i, 0] * SS), int(q[i, 1] * SS))
            b = (int(q[i + 1, 0] * SS), int(q[i + 1, 1] * SS))
            th = max(1, int(round(w[i] * SS)))
            cv2.line(self.layer, a, b, 255, th, cv2.LINE_AA)

    def dot(self, x, y, r):
        cv2.circle(self.layer, (int(x * SS), int(y * SS)), max(1, int(r * SS)), 255, -1, cv2.LINE_AA)

    def alpha(self):
        a = cv2.resize(self.layer, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return a

    def apply(self, cv, col=(0.27, 0.17, 0.10), op=0.9, mask=None):
        a = self.alpha() * op
        if VIGNETTE is not None:
            a = a * np.clip(VIGNETTE * 1.25, 0, 1)
        if INK_REACH is not None:
            a = a * INK_REACH
        if mask is not None:
            a = a * mask
        # the ink soaks into the paper: slightly grainy
        a = a * (1 - 0.18 * np.clip(PAPER_HF, 0, 1))
        col = np.asarray(col, np.float32)[None, None, :]
        cv[:] = cv * (1 - a[..., None]) + col * a[..., None]
        return cv


def hatch(ink, region, angle, spacing, length=None, w=1.0, jitter=0.6, seed=0, density=None, breaks=0.4):
    """Hatching inside a region. `density` (0..1) decides which lines are drawn."""
    r = np.random.default_rng(seed)
    ys, xs = np.nonzero(region > 0.5)
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ca, sa = np.cos(angle), np.sin(angle)
    diag = np.hypot(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = -diag / 2
    while k < diag / 2:
        # a line running at the angle `angle`, offset by k along the normal
        ox, oy = cx - sa * k, cy + ca * k
        ts = np.arange(-diag / 2, diag / 2, 2.0)
        px = ox + ca * ts
        py = oy + sa * ts
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        val = np.zeros_like(px)
        val[inside] = region[py[inside].astype(int), px[inside].astype(int)]
        if density is not None:
            dv = np.zeros_like(px)
            dv[inside] = density[py[inside].astype(int), px[inside].astype(int)]
            val = val * (dv > r.random() * 0.9 + 0.05)
        on = val > 0.5
        # runs
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                if j - i > 3:
                    seg = np.stack([px[i:j], py[i:j]], 1)
                    if length is not None and len(seg) * 2 > length:
                        # chop into shorter strokes
                        st = 0
                        while st < len(seg) - 2:
                            ln = int(r.uniform(0.6, 1.2) * length / 2)
                            ink.line(seg[st:st + ln], w0=w, jitter=jitter, taper=0.3, breaks=breaks, pressure=0.3)
                            st += ln + r.integers(1, 5)
                    else:
                        ink.line(seg, w0=w, jitter=jitter, taper=0.3, breaks=breaks, pressure=0.3)
                i = j
            else:
                i += 1
        k += spacing * r.uniform(0.8, 1.2)


class Pencil:
    """Colored pencil: a thin, grainy line that catches on the high spots of the paper."""

    def __init__(self):
        self.layers = {}

    def stroke(self, key, pts, w=1.0):
        if key not in self.layers:
            self.layers[key] = np.zeros((H * 2, W * 2), np.uint8)
        L = self.layers[key]
        p = np.round(np.asarray(pts) * 2 * 8).astype(np.int32)
        cv2.polylines(L, [p], False, 255, max(1, int(round(w * 2))), cv2.LINE_AA, 3)

    def apply(self, cv, key, col, op=0.7, mode='multiply', grain=0.15):
        if key not in self.layers:
            return cv
        a = cv2.resize(self.layers[key], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        a = a * np.clip((PAPER_HF + 1) / 2 + grain, 0, 1) ** 1.5 * op
        if VIGNETTE is not None:
            a = a * VIGNETTE
        if INK_REACH is not None:
            a = a * INK_REACH
        col = np.asarray(col, np.float32)[None, None, :]
        if mode == 'multiply':
            cv[:] = cv * (1 - a[..., None] * (1 - col))
        else:
            cv[:] = cv * (1 - a[..., None]) + col * a[..., None]
        return cv


def save(cv, path, q=93):
    img = (np.clip(cv, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if path.endswith('.jpg'):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        cv2.imwrite(path, img)
