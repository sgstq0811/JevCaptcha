"""Skill-library Jev agent for Open CaptchaWorld. The agent is not told the puzzle type. Clef looks at the numbered
screenshot, the instruction and a summary of the page's elements (button texts, number of pictures, tile grids, text
boxes, draggables) and picks one skill from a library of interaction patterns; the skill then solves the
puzzle with typed decisions. Perception is generic (screenshots, DOM elements found by kind, same-size groups, the
pixels that change when a control is clicked); questions are built from the instruction text, never from a type name.

  cycle  : a control cycles part of the puzzle through states -> the picture that changes gives the states, an unchanged
           picture the reference; Clef picks the attribute to match (direction / number / object / position); score =
           Noul(reference, state); a number: number match x 'the state matches the reference's number n' check; direction is matched alone, 8 equally turned states read jointly (cyclic group)
  select : a grid of tiles -> Noul per tile (the tile, the instruction quoted in the question); one tile: the argmax,
           all that fit: the tiles above the largest gap of the sorted log-odds
  swap   : a grid of tiles, swap two to line up three identical ones -> pairwise 'same picture' Noul, best line
           (a row or column counts before a diagonal: diagonal lines score half)
  point  : click one point on a picture -> if Clef says the place is the end of a drawn line, the two ends of the most
           colourful line are the candidates; if it is an area enclosed by drawn lines picked by size, the regions left
           after closing the (dashed) lines give the areas and their sizes; otherwise dot render-verify on a lattice, coarse to fine; each dot scores
           P(right place) x (1 - P(on something the picture marks as not to be clicked))
  order  : click items in the order of a reference strip -> count the strip's items, locate each one
  drag   : drag a piece into place -> positions where the piece's outline meets the picture's edges and its own picture
           matches the board (brightness-invariant) are the candidates, drag-watch over them, then correct the drop
           until the piece sits there
  hold   : press and hold -> release once the puzzle has changed and settled
  type   : a number answer -> decade, then digit
After the skill the agent clicks the element Clef picks as the submit control, unless the page already submitted."""
import os, io, sys, json, argparse, itertools
import numpy as np, cv2
from PIL import Image
from PIL import ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from ocw_generic import GenericAgent, FIND, dedupe, groups, changed_box, run
import math
from strokes import line_ends, line_end_cands
from regions import regions
from ocw_policy import NUM, LINES3, DIRS, choice, argmax, match
ABL = set(filter(None, os.environ.get("ABL", "").split(",")))   # ablation: components switched off

SKILLS = {
    "cycle": "buttons (arrows) change a picture through several states; choose the state that matches the reference or the instruction",
    "select": "a grid of tiles or squares; select the tiles that fit the instruction",
    "swap": "a grid of tiles; swap two tiles so that identical tiles line up",
    "point": "click one point on a picture",
    "order": "click several items on a picture in the order given by a reference",
    "drag": "drag a piece to the right place",
    "hold": "press and hold a button",
    "type": "type a number answer into a text box",
}


ATTRS = {"direction": "the direction it faces or points", "object": "the same object or picture",
         "number": "the same number or total", "position": "the same position or place"}


def box_of(e, box):
    return (round(e["x"] - box["x"]), round(e["y"] - box["y"]), round(e["x"] - box["x"] + e["w"]), round(e["y"] - box["y"] + e["h"]))


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def fit_map(bd, rgba, pw, ph):
    """score map over the board (RGB array) for the top-left of a piece (RGBA array) shown pw x ph (+-10%): its outline
    against the board's edges plus its own picture against the board (brightness-invariant); (map, w, h) or None"""
    g = cv2.cvtColor(bd, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    gm /= gm.mean() + 1e-6
    best = None
    for s in (.9, .95, 1, 1.05, 1.1):
        w, h = max(4, round(pw * s)), max(4, round(ph * s))
        if w >= gm.shape[1] or h >= gm.shape[0]:
            continue
        m = (cv2.resize(rgba[..., 3], (w, h), interpolation=cv2.INTER_AREA) > 128).astype(np.uint8)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        r = cv2.matchTemplate(gm, edge, cv2.TM_CCORR) / max(1, edge.sum())
        pr = cv2.resize(rgba[..., :3], (w, h), interpolation=cv2.INTER_AREA)       # the piece's own picture vs the board,
        cc = cv2.matchTemplate(bd, np.ascontiguousarray(pr), cv2.TM_CCOEFF_NORMED, mask=np.repeat(m[..., None], 3, 2))  # brightness-invariant
        r = r / (r.max() + 1e-6) + (0 if "content" in ABL else np.nan_to_num(np.clip(cc, 0, 1)))
        if best is None or r.max() > best[0].max():
            best = (r, w, h)
    return best


def pick_all(p):
    """'all tiles that fit': the tiles above the largest gap of the sorted log-odds (the fitting tiles form the upper
    mode even when their level is below 0.5)"""
    if "gap" in ABL:
        return [k for k, v in enumerate(p) if v > 0.5] or [argmax(p)]
    lo = [math.log(min(max(v, 1e-6), 1 - 1e-6) / (1 - min(max(v, 1e-6), 1 - 1e-6))) for v in p]
    o = sorted(range(len(p)), key=lambda k: -lo[k])
    n = 1 + max(range(len(o) - 1), key=lambda i: lo[o[i]] - lo[o[i + 1]]) if len(o) > 1 else 1
    return sorted(o[:n])


WHOLE = "Does this picture look like one whole, natural picture, the right way up, with every part in its right place?"


class SkillAgent(GenericAgent):
    def ask(self, st, ims, instr, opts):
        p = self.clef.decide([self.pol.rec(st, ims, {"a": {"type": "choice", "instructions": instr, "criteria": opts}})])[0]["a"]
        return max(p, key=p.get)

    def look(self, pg):
        im, box = self.shot(pg)
        return im, box, dedupe(pg.evaluate(FIND))

    def click(self, pg, e):
        pg.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
        pg.wait_for_timeout(200)

    def submit(self, pg, st, verdict):
        if verdict:
            return "page submitted"
        im, box, els = self.look(pg)
        tiles = {i for g in groups(els) for i in g}
        used = None if "used" in ABL else getattr(self, "used", None)   # never the control the skill itself just worked
        same = lambda e: used and abs(e["x"] - used["x"]) < 3 and abs(e["y"] - used["y"]) < 3 and abs(e["w"] - used["w"]) < 3
        cand = {str(i): f"element {i} ('{e['text']}')" for i, e in enumerate(els, 1) if len(e["text"]) > 1 and not e["input"] and i not in tiles and not same(e)}
        if not cand:
            return "no submit control"
        k = self.ask(st, [self.marked(im, box, els)], "Which element submits the answer?", cand)
        self.click(pg, els[int(k) - 1])
        return f"submit {k}"

    def solve(self, pg, prompt, verdict):
        st = {"instruction": prompt}
        self.used = None
        im, box, els = self.look(pg)
        gs = groups(els)
        aff = {"buttons": [e["text"] or "(no text)" for e in self.controls(els)][:12], "pictures": len(self.bigs(els)),
               "grids_of_same_size_tiles": [len(g) for g in gs], "text_boxes": sum(e["input"] for e in els),
               "draggable": sum(e["drag"] and not e["big"] for e in els)}
        sk = self.ask({**st, "page_elements": aff}, [self.marked(im, box, els)], "You are solving the puzzle in the image (interactive elements are outlined and numbered). "
                      "Which kind of interaction does it need?", SKILLS)
        log = [f"skill {sk}"]
        log.append(getattr(self, "sk_" + sk)(pg, st, im, box, els))
        pg.wait_for_timeout(300)
        log.append(self.submit(pg, st, verdict))
        return log

    # ------------------------------------------------------------------ skills
    def controls(self, els):
        return [e for e in els if not e["input"] and not e["big"]]

    def bigs(self, els):
        return sorted([e for e in els if e["big"]], key=lambda e: -e["w"] * e["h"])

    def sk_cycle(self, pg, st, im, box, els):
        for e in self.controls(els)[:4]:                # the first control that cycles part of the puzzle
            looks, keys = [], []
            for _ in range(16):
                s, _ = self.shot(pg)
                s = s.resize(im.size)
                key = s.resize((96, 96)).convert("L").tobytes()
                if any(max(abs(a - b) for a, b in zip(key, k)) < 24 for k in keys):
                    break
                looks.append(s); keys.append(key)
                self.click(pg, e)
            if len(looks) >= 2:
                break
        else:
            return "no control changes the puzzle"
        z = changed_box(looks)
        za = (z[2] - z[0]) * (z[3] - z[1])
        bb = [box_of(b, box) for b in self.bigs(els)]
        fr = [self.changes(looks, b) for b in bb]
        ref = []
        if fr and max(fr) > 0.002:                     # the picture that changes most holds the states; the one that
            k = argmax(fr)                             # changes least (layout shifts move it a little) is the reference
            z = bb[k]
            rest = [i for i in range(len(bb)) if i != k and overlap(bb[i], z) < 0.2 * za]
            ref = [bb[min(rest, key=lambda i: fr[i])]] if rest else []
        else:
            ref = [b for b in bb if overlap(b, z) < 0.2 * za]
        if ref:
            r = looks[0].crop(ref[0])
        else:
            r = looks[0].copy()
            ImageDraw.Draw(r).rectangle(z, fill=(128, 128, 128))
        cands = [s.crop(z) for s in looks]
        if os.environ.get("SKILL_DUMP"):              # debugging: keep what the cycle skill compares
            d = os.environ["SKILL_DUMP"]
            os.makedirs(d, exist_ok=True)
            n = len(os.listdir(d))
            r.save(f"{d}/{n:04d}_ref.png")
            for i, c in enumerate(cands):
                c.save(f"{d}/{n:04d}_{i}.png")
        attr, k, p = self.choose(st, looks[0], r, cands)
        for _ in range(k):                             # back at look 0 after the cycle
            self.click(pg, e)
        return f"cycle ({attr}): {len(looks)} states, chose {k + 1} (P={p[k]:.2f})"

    def choose(self, st, scene, r, cands):
        """which state matches the reference: Clef picks the attribute to match, then the matching rule for it;
        without a reference, the state that Clef sees as one whole, natural picture"""
        if r is None:
            q = f"The instruction is: '{st['instruction']}'. Is it already done in this picture?" if "noref" in ABL else \
                "Does this picture look like one whole, natural picture, with every part in its right place?" if "upright" in ABL else \
                WHOLE
            p = self.noul(st, [[c] for c in cands], q)
            return "instruction", argmax(p), p
        attr = self.ask(st, [scene], "The buttons change part of this puzzle. What must the changing part match?", ATTRS)
        if attr == "direction":                        # the turned states are read jointly; the direct check adds a bias
            p = self.match_direction(st, r, cands)
            z = sum(p) or 1
            p = [v / z for v in p]
        else:
            p = self.match_same(st, r, cands)
        if attr == "number" and "numver" in ABL:       # product of the direct check and the number match
            q = self.match_number(st, r, cands)
            p = [a * b for a, b in zip(p, q)]
            z = sum(p) or 1
            p = [v / z for v in p]
        elif attr == "number":                         # number match x a check of 'matches the reference's number n'
            p = self.verify_number(st, r, cands)
        k = argmax(p)
        return attr, k, p

    def changes(self, looks, b):
        """largest fraction of the box's pixels that differ from the first look"""
        import numpy as np
        a = [np.asarray(l.crop(b).convert("L"), dtype=np.int16) for l in looks]
        return max((np.abs(x - a[0]) > 30).mean() for x in a[1:])

    def match_same(self, st, r, cands):
        return self.noul(st, [[r, c] for c in cands], "Image 1 is the reference part of the puzzle. Image 2 is one state of the part the buttons change. "
                                                      "Does image 2 correctly answer the instruction given the reference?")

    match_object = match_position = match_same

    def read(self, st, ims, q):
        return [d["v"] for d in self.clef.decide([self.pol.rec(st, [im], {"v": q}) for im in ims])]

    def match_direction(self, st, r, cands):
        """read the direction of the reference and of every state; when the states are one object turned in 8 equal steps,
        estimate its heading from all states jointly (cyclic-group consistency) before matching"""
        q = choice("In which direction does the object in the image point or face?" if "front" in ABL else
                   "Which way is the object in the image pointing or facing (where its front or tip is)?", DIRS)
        pr = self.read(st, [r], q)[0]
        if len(cands) != 8 and "dirsep" not in ABL:     # separate pictures: each read like the reference
            return match(pr, self.read(st, cands, q))
        cs = self.read(st, cands, choice("This object is turned by the buttons. In which direction does the top of the object (its head or upper side) point?", DIRS))
        if len(cs) != 8:
            return match(pr, cs)
        res = []
        for step in (1, -1):                           # either turning sense
            base = [sum(math.log(cs[i][DIRS[(d0 + step * i) % 8]] + 1e-6) for i in range(8)) for d0 in range(8)]
            res.append((max(base), base, step))
        if "margs" in ABL:                             # variant: marginalise the turning sense instead of plugging in its MAP
            m = max(max(b) for _, b, _ in res)
            w = [[math.exp(x - m) for x in b] for _, b, _ in res]
            z = sum(map(sum, w))
            return [sum(w[j][d0] / z * pr[DIRS[(d0 + st_ * i) % 8]] for j, (_, _, st_) in enumerate(res) for d0 in range(8)) for i in range(8)]
        _, base, step = max(res)
        m = max(base)
        pb = [math.exp(x - m) for x in base]
        z = sum(pb)
        return [sum(pb[d0] / z * pr[DIRS[(d0 + step * i) % 8]] for d0 in range(8)) for i in range(8)]

    def match_number(self, st, r, cands):
        q = choice("What number does the image show or add up to?", NUM(100))
        pr, cs = self.read(st, [r], q)[0], self.read(st, cands, q)
        return match(pr, cs)

    VN = ("The instruction is: '{i}'. Image 2 is the reference part of the puzzle; its number is {n}. Is image 1 the state that matches the reference?",)

    def verify_number(self, st, r, cands, top=2):
        """the reference's number n (top 2 readings); each state scores match(numbers read) x
        sum_n P_ref(n) P(state matches the reference whose number is n | state, reference)"""
        q = choice("What number does the image show or add up to?", NUM(100))
        pr, cs = self.read(st, [r], q)[0], self.read(st, cands, q)
        p = match(pr, cs)
        ns = sorted(pr, key=pr.get, reverse=True)[:top]
        for v in self.VN:
            sc = [0.0] * len(cands)
            for n in ns:
                sc = [s + pr[n] * x for s, x in zip(sc, self.noul(st, [[c, r] for c in cands], v.format(i=st["instruction"], n=n)))]
            p = [a * b for a, b in zip(p, sc)]
        z = sum(p) or 1
        return [v / z for v in p]

    def tiles(self, els):
        gs = groups(els)
        return max(gs, key=len) if gs else []

    def tile_p(self, st, tiles, example=None):
        """P(select) per tile, the instruction quoted in the question; a puzzle's example picture comes along as image 2"""
        q = f"The instruction is: '{st['instruction']}'. "
        if "up" not in ABL:                             # small tiles are shown 3x larger
            tiles = [t.resize((3 * t.size[0], 3 * t.size[1]), Image.LANCZOS) if t.size[0] < 200 else t for t in tiles]
        if example is not None:
            return self.noul(st, [[t, example] for t in tiles], q + "Image 2 is the example the puzzle shows. Does image 1 fit the instruction?")
        return self.noul(st, [[t] for t in tiles], q + "Does this picture fit the instruction?")

    def tile_scores(self, pg, st, im, box, els, g):
        return self.tile_p(st, [im.crop(box_of(els[i - 1], box)) for i in g])

    def sk_select(self, pg, st, im, box, els):
        g = self.tiles(els)
        if not g:
            return "no tiles"
        p = self.tile_scores(pg, st, im, box, els, g)
        one = self.ask(st, [im], "Does the instruction ask for exactly one tile or for all tiles that fit?", {"one": "exactly one", "all": "all that fit"})
        sel = [argmax(p)] if one == "one" else pick_all(p)
        for k in sel:
            self.click(pg, els[g[k] - 1])
        return f"select tiles {[g[k] for k in sel]}"

    def sk_swap(self, pg, st, im, box, els):
        g = self.tiles(els)
        if len(g) != 9:
            return self.sk_select(pg, st, im, box, els)
        cr = [im.crop(box_of(els[i - 1], box)) for i in g]
        pairs = list(itertools.combinations(range(9), 2))
        p = self.noul({"task": "Compare two tiles of a picture grid."}, [[cr[i], cr[j]] for i, j in pairs],
                      "Do image 1 and image 2 show the same picture (the same object or animal in the same style)?")
        same = [[1.0] * 9 for _ in range(9)]
        for (i, j), v in zip(pairs, p):
            same[i][j] = same[j][i] = v
        best, arg = -1, (0, 1)
        for i, j in pairs:
            pos = list(range(9))
            pos[i], pos[j] = pos[j], pos[i]
            for m, ln in enumerate(LINES3):
                a, b, c = (pos[k] for k in ln)
                s = same[a][b] * same[b][c] * same[a][c] * (1 if m < 6 or "axis" in ABL else .5)   # a row or column before a diagonal
                if s > best:
                    best, arg = s, (i, j)
        for k in arg:
            self.click(pg, els[g[k] - 1])
        return f"swap tiles {g[arg[0]]} and {g[arg[1]]}"

    def dot(self, img):
        def render(x, y):
            o = img.copy()
            r = max(6, img.size[0] // 60)
            d = ImageDraw.Draw(o)
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(255, 255, 255))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
            return o
        return render

    OK = "A red dot with a white ring has been placed on the picture. Is the red dot at a place that the instruction says to click?"
    BAD = "A red dot with a white ring has been placed on the picture. Is the red dot on something the picture marks as not to be clicked (for example labelled 'don't click')?"
    ON = "A red dot with a white ring has been placed on the picture. Is the red dot on an object, rather than on empty background?"

    LINE = "Is the place to click the end of a line, route or path drawn in the picture?"
    AREA = "Does the instruction ask to click inside one of the areas enclosed by lines drawn on the picture?"
    RULES = {"largest": "the largest area", "smallest": "the smallest area", "content": "by what the area shows or by some other property"}
    OKZ = "A red dot with a white ring has been placed on the picture; image 2 zooms in around the dot. Is the red dot at the place that the instruction says to click?"
    STOP = "A red dot with a white ring has been placed on the picture; image 2 zooms in around the dot. Does the drawn line stop at the dot, without going on past it?"

    def safe_point(self, img, st, coarse=10, fine=5, topk=3):
        """dot render-verify; each point scores P(right place) x (1 - P(marked as not to click)) x P(on an object); coarse lattice, then finer"""
        W, H = img.size
        ren = self.dot(img)

        def score(pts):
            ims = [[ren(x, y)] for x, y in pts]
            ok, bad = self.noul(st, ims, self.OK), self.noul(st, ims, self.BAD)
            on = [1.0] * len(pts) if "obj" in ABL else self.noul(st, ims, self.ON)   # a click lands on something
            return [a * (1 - b) * c for a, b, c in zip(ok, bad, on)]
        pts = [((i + .5) * W / coarse, (j + .5) * H / coarse) for j in range(coarse) for i in range(coarse)]
        p = score(pts)
        fp = []
        for k in sorted(range(len(p)), key=lambda k: -p[k])[:topk]:
            for j in range(fine):
                for i in range(fine):
                    fp.append((pts[k][0] + (i - (fine - 1) / 2) * W / coarse / fine, pts[k][1] + (j - (fine - 1) / 2) * H / coarse / fine))
        pf = score(fp)
        return fp[argmax(pf)]

    def point_in(self, st, img):
        """the point to click in a picture"""
        ends = [] if "line" in ABL or self.noul(st, [[img]], self.LINE)[0] <= .5 else line_ends(img) if "ends" in ABL else line_end_cands(img)
        if ends:                                        # the place is an end of a drawn line, which may be broken where things hide it
            W, H = img.size
            ren, z = self.dot(img), W // 5
            ims = [[o := ren(x, y), o.crop((int(max(0, x - z)), int(max(0, y - z)), int(min(W, x + z)), int(min(H, y + z))))] for x, y in ends]
            x, y = ends[argmax(self.noul(st, ims, self.OKZ if "ends" in ABL else self.STOP))]
        elif "area" not in ABL and self.noul(st, [[img]], self.AREA)[0] > .5 and (rule := self.ask(st, [img], "How does the instruction pick the area to click?", self.RULES)) != "content" \
                and (rs := regions(img)):                # an area enclosed by drawn lines, picked by its size
            (x, y), _ = (max if rule == "largest" else min)(rs, key=lambda r: r[1])
        else:
            x, y = self.safe_point(img, st)
        return x, y

    def sk_point(self, pg, st, im, box, els):
        bs = self.bigs(els)
        e = bs[0] if bs else {"x": box["x"], "y": box["y"], "w": box["width"], "h": box["height"]}
        img = im.crop(box_of(e, box))
        x, y = self.point_in(st, img)
        pg.mouse.click(e["x"] + x * e["w"] / img.size[0], e["y"] + y * e["h"] / img.size[1])
        return f"point ({x / img.size[0]:.2f}, {y / img.size[1]:.2f})"

    def sk_order(self, pg, st, im, box, els):
        bs = self.bigs(els)
        if len(bs) < 2:
            return self.sk_point(pg, st, im, box, els)
        main, ref = bs[0], bs[1]
        order, img = im.crop(box_of(ref, box)), im.crop(box_of(main, box))
        k = int(self.ask(st, [order], "How many separate items are shown in this reference?", {str(i): str(i) for i in range(2, 9)}))
        W, H = order.size
        horiz = W >= H
        items = [order.crop((i * W // k, 0, (i + 1) * W // k, H) if horiz else (0, i * H // k, W, (i + 1) * H // k)) for i in range(k)]
        q = "The image is divided into numbered red cells. Image 2 shows one item. Which numbered cell contains that item in image 1?"
        xy = [self.pol.locate(img, st, q, extra=(it,)) for it in items] if "assign" in ABL else self.pol.locate_all(img, st, q, items)
        pts = []
        for x, y in xy:
            pg.mouse.click(main["x"] + x * main["w"] / img.size[0], main["y"] + y * main["h"] / img.size[1])
            pg.wait_for_timeout(200)
            pts.append((round(x / img.size[0], 2), round(y / img.size[1], 2)))
        return f"order {pts}"

    PIECE = """([x, y]) => { for (const e of document.elementsFromPoint(x, y)) if (e.tagName === 'IMG') {
        const b = e.getBoundingClientRect(); return {src: e.src, x: b.x, y: b.y, w: b.width, h: b.height}; } return null; }"""

    def fits(self, pg, e, area, k=5):
        """where the dragged piece's outline lines up with edges of the picture: outline-vs-edge correlation over the
        picture at the piece's on-screen size (+-10%), the k best separated positions (page coordinates of the piece centre)"""
        pc = pg.evaluate(self.PIECE, [e["x"] + e["w"] / 2, e["y"] + e["h"] / 2])
        if not pc:
            return []
        rgba = np.array(Image.open(io.BytesIO(pg.request.get(pc["src"]).body())).convert("RGBA"))
        if rgba[..., 3].min() > 128:                    # opaque piece: its outline is its box
            rgba[[0, -1], :, 3] = rgba[:, [0, -1], 3] = 0
        im, box = self.shot(pg)
        bd = np.array(im.crop(box_of(area, box)))
        best = fit_map(bd, rgba, pc["w"], pc["h"])
        if best is None:
            return []
        r, w, h = best
        px, py = round(pc["x"] - area["x"] + (pc["w"] - w) / 2), round(pc["y"] - area["y"] + (pc["h"] - h) / 2)
        r[max(0, py - h):max(0, py + h + 1), max(0, px - w):max(0, px + w + 1)] = -1   # not where the piece already lies
        q = 6                                           # nor against the picture's own border
        r[:q], r[-q:], r[:, :q], r[:, -q:] = -1, -1, -1, -1
        out = []
        for _ in range(k):
            _, mx, _, (x, y) = cv2.minMaxLoc(r)
            out.append((area["x"] + x + w / 2, area["y"] + y + h / 2, mx))
            r[max(0, y - h // 2):y + h // 2 + 1, max(0, x - w // 2):x + w // 2 + 1] = -1
        return out

    def sk_drag(self, pg, st, im, box, els):
        ds = [e for e in els if e["drag"]] or self.controls(els)
        if not ds:
            return "nothing to drag"
        e = min(ds, key=lambda e: e["w"] * e["h"])
        bs = [b for b in self.bigs(els) if b["w"] * b["h"] > 4 * e["w"] * e["h"]]     # the picture the piece goes into
        cands = self.fits(pg, e, bs[0]) if bs else []
        if not cands:
            return "drag " + self.drops(pg, st, e, bs[0] if bs else None)
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2       # hold the piece, look at it over each fit, release at the best
        src = (pg.evaluate(self.PIECE, [cx, cy]) or {}).get("src")
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        ims = []
        for x, y, _ in cands:
            pg.mouse.move(x, y, steps=8)
            pg.wait_for_timeout(80)
            ims.append([self.shot(pg)[0]])
        p = self.noul(st, ims, "The picture shows the puzzle while a piece is being dragged. If it is released here, does the puzzle satisfy the instruction?")
        fm = cands[0][2]
        sc = [q * (f / fm) ** 4 for q, (_, _, f) in zip(p, cands)]
        b = argmax(sc)
        tx, ty = cands[b][0], cands[b][1]
        pg.mouse.move(tx, ty, steps=8)
        pg.wait_for_timeout(100)
        pg.mouse.up()
        pc = None
        for _ in range(0 if "loop" in ABL else 3):      # closed loop: find where the piece really sits, drag it by the rest
            pc = pg.evaluate("(s) => { for (const e of document.querySelectorAll('#puzzle-container img')) if (e.src === s) {"
                             " const b = e.getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2]; } return null; }", src)
            if not pc or abs(pc[0] - tx) < 1.5 and abs(pc[1] - ty) < 1.5:
                break
            pg.mouse.move(*pc)
            pg.mouse.down()
            pg.mouse.move(pc[0] + (tx - pc[0]), pc[1] + (ty - pc[1]), steps=12)
            pg.wait_for_timeout(100)
            pg.mouse.up()
        return f"drag fit {b} of {len(cands)} (P={[round(q, 2) for q in p]}, fit={[round(f / fm, 2) for _, _, f in cands]}) piece at {pc}"

    def sk_hold(self, pg, st, im, box, els):
        cs = [e for e in els if e["tag"] == "button" or e["cursor"] == "pointer"] or self.controls(els)
        k = self.ask(st, [self.marked(im, box, els)], "Which element should be pressed and held?", {str(els.index(e) + 1): f"element {els.index(e) + 1}" for e in cs})
        self.used = els[int(k) - 1]
        return self.hold(pg, self.used)

    def sk_type(self, pg, st, im, box, els):
        ins = [e for e in els if e["input"]]
        if not ins:
            return "no text box"
        v = self.number(st, im)
        self.click(pg, ins[0])
        pg.keyboard.press("Control+A")
        pg.keyboard.type(v)
        return f"type {v}"


class Episode:
    """adapter so ocw_generic.run drives one skill episode: the first step solves, later steps end the episode"""
    def __init__(self, agent):
        self.agent = agent
        self.verdict = None

    def step(self, pg, prompt, hist):
        if hist:
            return None
        return " | ".join(self.agent.solve(pg, prompt, self.verdict))


if __name__ == "__main__":
    from ocw_agent import TYPES
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Episode(SkillAgent(a.model, True)), a.types, a.out, a.limit)
