"""Regions of a picture enclosed by drawn (possibly dashed) lines: the lines are a rare colour cluster, closed with a
dilation that bridges the dashes; the regions are the connected parts of what is left."""
import numpy as np, cv2


def regions(img, k=6, side=256, gap=.03):
    """list of (centre_xy in picture pixels, area fraction) for each enclosed region, per line-colour cluster tried;
    returns the partition with the most regions of >= 2% area"""
    a = np.array(img.convert('RGB').resize((side, round(side * img.size[1] / img.size[0])), Image_LANCZOS))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    _, lb, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    lb = lb.reshape(a.shape[:2])
    f = img.size[0] / side
    best = []
    for c in range(k):
        m = (lb == c).astype(np.uint8)
        if not .005 <= m.mean() <= .2:
            continue
        r = max(1, round(gap * side / 2))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, cl, st, _ = cv2.connectedComponentsWithStats(1 - m, 4)
        out = []
        for i in range(1, n):
            if st[i, 4] < .02 * m.size:
                continue
            dt = cv2.distanceTransform((cl == i).astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(dt.argmax(), dt.shape)
            out.append(((x * f, y * f), st[i, 4] / m.size))
        if len(out) > len(best):
            best = out
    return best


from PIL import Image
Image_LANCZOS = Image.LANCZOS
