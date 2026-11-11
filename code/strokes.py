"""Strokes drawn in a rare colour and their ends (geodesic far points)."""
import json, sys, os
import numpy as np, cv2
from PIL import Image
from collections import deque



def bfs(m, s):
    d = np.full(m.shape, -1, np.int32)
    d[s] = 0
    q = deque([s])
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            v = (y + dy, x + dx)
            if 0 <= v[0] < m.shape[0] and 0 <= v[1] < m.shape[1] and m[v] and d[v] < 0:
                d[v] = d[y, x] + 1
                q.append(v)
    return d


def far_ends(m, side=200):
    """the two ends of a stroke: farthest pixel from any pixel, then the farthest pixel from that one (geodesic)"""
    f = min(1, side / max(m.shape))
    s = cv2.resize(m.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(s)
    if not len(ys):
        return []
    d = bfs(s, (ys[0], xs[0]))
    a = np.unravel_index(d.argmax(), d.shape)
    d = bfs(s, a)
    b = np.unravel_index(d.argmax(), d.shape)
    return [(a[1] / f, a[0] / f), (b[1] / f, b[0] / f)]


def strokes(img, k=8, side=200):
    """rare-colour strokes: list of (mask at picture size, [end_a, end_b])"""
    a = np.array(img.convert('RGB').resize((side, round(side * img.size[1] / img.size[0]))))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    _, lb, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    lb = lb.reshape(a.shape[:2])
    f = img.size[0] / side
    out = []
    for c in range(k):
        m = (lb == c).astype(np.uint8)
        if not .003 <= m.mean() <= .15:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, cl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in sorted(range(1, n), key=lambda i: -st[i, 4])[:2]:
            if st[i, 4] >= .002 * m.size:
                big = cv2.resize((cl == i).astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST)
                out.append((big, [(x * f, y * f) for x, y in far_ends(cl == i, side=10**9)]))
    return out


def stroke_ends(img, k=8, side=200):
    """ends of strokes drawn in a rare colour: k-means colour clusters covering 0.3-15% of the picture, their largest
    connected parts, the two geodesic ends of each"""
    a = np.array(img.convert('RGB').resize((side, round(side * img.size[1] / img.size[0]))))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    _, lb, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    lb = lb.reshape(a.shape[:2])
    f = img.size[0] / side
    out = []
    for c in range(k):
        m = (lb == c).astype(np.uint8)
        if not .003 <= m.mean() <= .15:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, cl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in sorted(range(1, n), key=lambda i: -st[i, 4])[:2]:
            if st[i, 4] >= .002 * m.size:
                out += [(x * f, y * f) for x, y in far_ends(cl == i, side=10**9)]
    return out


def line_ends(img):
    """the two ends of the most colourful line: strokes whose mean chroma is within 15% of the most colourful one,
    joined, then the geodesic far ends of the joined mask"""
    lab = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2LAB).astype(np.float32)
    ch = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    sk = [(float(ch[m > 0].mean()), m) for m, _ in strokes(img)]
    if not sk:
        return []
    cm = max(c for c, _ in sk)
    m = np.zeros(ch.shape, np.uint8)
    for c, mk in sk:
        if c >= .85 * cm:
            m |= mk.astype(np.uint8)
    m = cv2.dilate(m, np.ones((max(3, img.size[0] // 40),) * 2, np.uint8))
    n, cl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    big = 1 + int(np.argmax(st[1:, 4])) if n > 1 else 0
    if not big:
        return []
    orig = np.zeros_like(m)
    for c, mk in sk:
        if c >= .85 * cm:
            orig |= mk.astype(np.uint8)
    ys, xs = np.nonzero(orig)
    out = []
    for x, y in far_ends(cl == big):                   # pull each end back onto the undilated line, half a width in
        d = np.hypot(xs - x, ys - y)
        near = d < d.min() + img.size[0] / 40
        out.append((float(xs[near].mean()), float(ys[near].mean())))
    return out


def line_mask(img):
    """every piece of the most colourful line (a line hidden in places by other things breaks into pieces)"""
    lab = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2LAB).astype(np.float32)
    ch = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    m = (ch >= .6 * np.percentile(ch, 99.5)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def line_end_cands(img, radii=(8, 16, 32, 48)):
    """candidate ends of a broken line: its pieces joined across gaps up to r px, for several r; the geodesic far ends
    of the largest joined part, pulled back onto the drawn pixels"""
    m = line_mask(img)
    out = []
    for r in radii:
        n, cl, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((r, r), np.uint8)), 8)
        if n < 2:
            continue
        big = 1 + int(np.argmax(st[1:, 4]))
        ys, xs = np.nonzero(m & (cl == big))
        for x, y in far_ends(cl == big):
            d = np.hypot(xs - x, ys - y)
            near = d < d.min() + img.size[0] / 40
            q = (float(xs[near].mean()), float(ys[near].mean()))
            if all(np.hypot(q[0] - v[0], q[1] - v[1]) > 10 for v in out):
                out.append(q)
    return out
