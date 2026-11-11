"""Tests of the predictions in THEORY.md from the raw typed decisions saved by theory_data.py.

Rotation (Theorem 1 / Corollary 1): for k = 1..8 views, average over all k-subsets of the 8 states:
  base error   P[MAP base direction != true base]   (true base = 'up' for every object: all base images are upright)
  final error  of the group-consistent rule and of the independent per-state rule (both may pick any state)
  bound        (|V|-1) exp(-k mu^2 / 2B^2) with mu, B estimated from the per-view log-likelihood ratios
Bingo (Proposition 3): ground-truth tile identities from pixel equality of tiles; per-pair error delta of the
thresholded P(same); relational-rule failure rate vs the bound (3 + W) * delta."""
import sys, json, math, itertools
import numpy as np
from PIL import Image, ImageOps

DIRS = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
LINES3 = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
d = json.load(open(sys.argv[1]))
data_root = sys.argv[2] if len(sys.argv) > 2 else None
EPS = 1e-6

# ---------------------------------------------------------------- rotation
R = d["rotation"]
n = 8


def base_post(states, S):
    lp = np.array([sum(math.log(states[i][DIRS[(b + i) % n]] + EPS) for i in S) for b in range(n)])
    p = np.exp(lp - lp.max())
    return p / p.sum()


rows = []
llr = []
for k in range(1, n + 1):
    be = fe_c = fe_i = cnt = 0
    for x in R:
        ref = np.array([x["ref"][v] for v in DIRS])
        for S in itertools.combinations(range(n), k):
            pb = base_post(x["states"], S)
            be += pb.argmax() != 0
            sc = [sum(pb[b] * ref[(b + i) % n] for b in range(n)) for i in range(n)]
            fe_c += int(np.argmax(sc)) != x["gold"]
            # independent rule: inner product of reference and each observed state; unobserved states score 0
            si = [sum(ref[v] * x["states"][i][DIRS[v]] for v in range(n)) if i in S else -1 for i in range(n)]
            fe_i += int(np.argmax(si)) != x["gold"]
            cnt += 1
    rows.append((k, be / cnt, fe_c / cnt, fe_i / cnt))
for x in R:
    for i in range(n):
        for b in range(1, n):
            llr.append(math.log(x["states"][i][DIRS[i % n]] + EPS) - math.log(x["states"][i][DIRS[(b + i) % n]] + EPS))
llr = np.array(llr)
mu, B = llr.mean(), np.abs(llr).max()
ref_err = np.mean([max(x["ref"], key=x["ref"].get) != DIRS[x["gold"] % n] for x in R])
print(f"Rotation: {len(R)} puzzles; per-view LLR mean mu={mu:.3f}, range B={B:.2f}, reference reading error={ref_err:.3f}")
print(" k  base_err  bound(Hoeffding)  final_err_group  final_err_indep")
for k, be, fc, fi in rows:
    bound = min(1.0, (n - 1) * math.exp(-k * mu ** 2 / (2 * B ** 2)))
    print(f" {k}  {be:.3f}     {bound:.3f}             {fc:.3f}            {fi:.3f}")
# empirical exponential fit of base error
ks = np.array([r[0] for r in rows]); be = np.array([max(r[1], 1e-4) for r in rows])
slope = np.polyfit(ks, np.log(be), 1)[0]
print(f" fitted base error ~ exp({slope:.3f} k)")

# ---------------------------------------------------------------- bingo
if data_root:
    tot_pairs = err_pairs = fails = 0
    bounds = []
    for x in d["bingo"]:
        img = ImageOps.exif_transpose(Image.open(f"{data_root}/Bingo/{x['id']}")).convert("RGB")
        W, H = img.size
        tiles = [np.asarray(img.crop((c * W // 3, r * H // 3, (c + 1) * W // 3, (r + 1) * H // 3)).resize((64, 64)), dtype=np.float32) for r in range(3) for c in range(3)]
        same = {}
        for (i, j), p in zip(map(tuple, x["pairs"]), x["p"]):
            truth = np.abs(tiles[i] - tiles[j]).mean() < 8.0
            same[(i, j)] = same[(j, i)] = p
            tot_pairs += 1
            err_pairs += (p > 0.5) != truth
        # relational rule
        best, arg = -1, None
        W2 = W3 = 0
        lab = {}
        for i, j in itertools.combinations(range(9), 2):
            pos = list(range(9)); pos[i], pos[j] = pos[j], pos[i]
            for ln in LINES3:
                a, b, c = (pos[k] for k in ln)
                s = same[(a, b)] * same[(b, c)] * same[(a, c)]
                m = sum(np.abs(tiles[u] - tiles[v]).mean() >= 8.0 for u, v in ((a, b), (b, c), (a, c)))
                W2 += m == 2
                W3 += m == 3
                if s > best:
                    best, arg = s, [i, j]
        fails += not any(sorted(arg) == sorted(g) for g in x["gold"])
        bounds.append((W2, W3))
    delta = err_pairs / tot_pairs
    W2m, W3m = (float(np.mean([b[i] for b in bounds])) for i in (0, 1))
    bnd = 3 * delta + W2m * delta ** 2 + W3m * delta ** 3
    print(f"Bingo: {len(d['bingo'])} puzzles; pair error delta={delta:.4f}; relational failure={fails}/{len(d['bingo'])} "
          f"= {fails / len(d['bingo']):.3f}; mean W2={W2m:.0f}, W3={W3m:.0f}; bound 3d+W2 d^2+W3 d^3 = {min(1, bnd):.3f}")


# ---------------------------------------------------------------- calibration of the raw decisions
def ece(conf, corr, bins=10):
    conf, corr = np.array(conf), np.array(corr, dtype=float)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e


conf, corr = [], []
for x in R:
    for i in range(n):
        p = x["states"][i]
        top = max(p, key=p.get)
        conf.append(p[top]); corr.append(top == DIRS[i])
print(f"Calibration (rotation state directions, top-1): acc={np.mean(corr):.3f}, ECE={ece(conf, corr):.3f}")
if data_root:
    conf, corr = [], []
    for x in d["bingo"]:
        img = ImageOps.exif_transpose(Image.open(f"{data_root}/Bingo/{x['id']}")).convert("RGB")
        W, H = img.size
        tiles = [np.asarray(img.crop((c * W // 3, r * H // 3, (c + 1) * W // 3, (r + 1) * H // 3)).resize((64, 64)), dtype=np.float32) for r in range(3) for c in range(3)]
        for (i, j), p in zip(map(tuple, x["pairs"]), x["p"]):
            truth = np.abs(tiles[i] - tiles[j]).mean() < 8.0
            conf.append(max(p, 1 - p)); corr.append((p > 0.5) == truth)
    print(f"Calibration (Bingo pair equality, Noul): acc={np.mean(corr):.3f}, ECE={ece(conf, corr):.3f}")
