"""Decision policies for the Open CaptchaWorld browser agent. Every perception is a typed Clef decision
(Noul / choice over named options); the action is computed from those decisions.

  verify      : Noul on (reference, candidate state) for every state reachable with the arrows -> argmax
  typed match : choice question reads an attribute from the reference and from every candidate -> best match
  consistency : the arrow states of Rotation_Match are one object turned in equal steps; its head direction is
                estimated from all states jointly before matching the reference direction
  pairwise    : Bingo tiles are compared pairwise (Noul 'same picture'); the swap is found by exhaustive search
  cell select : Noul per grid cell, select P > 0.5 (argmax when none / single-choice widgets)
  numbered    : clicks are chosen coarse-to-fine among numbered cells drawn on the image (choice question)
"""
import os, math, itertools
from PIL import Image, ImageDraw, ImageFont
from clef import Clef

DIRS = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
LETTERS = [chr(65 + i) for i in range(26)]
NUM = lambda n: [str(i) for i in range(n)]
LINES3 = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
VERIFY = ("Image 1 is the reference shown on the left of the puzzle. Image 2 is one candidate state of the puzzle on the right. "
          "Does image 2 correctly solve the puzzle instruction given the reference?")
FONT = ImageFont.load_default(size=22)


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": {o: o for o in opts}}


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


TOP1 = "top1" in os.environ.get("ABL", "").split(",")   # variant: compare only the most likely readings


def match(pr, pcs):
    if TOP1:
        a = max(pr, key=pr.get)
        return [float(max(pc, key=pc.get) == a) for pc in pcs]
    return [sum(pr[v] * pc.get(v, 0) for v in pr) for pc in pcs]


def grid_boxes(W, H, rows, cols):
    return [(c * W // cols, r * H // rows, (c + 1) * W // cols, (r + 1) * H // rows) for r in range(rows) for c in range(cols)]


def numbered(img, n):
    im = img.resize((768, round(768 * img.size[1] / img.size[0])))
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    for i in range(1, n):
        d.line([(i * W / n, 0), (i * W / n, H)], fill=(255, 0, 0, 200), width=2)
        d.line([(0, i * H / n), (W, i * H / n)], fill=(255, 0, 0, 200), width=2)
    for r in range(n):
        for c in range(n):
            x, y = c * W / n + 4, r * H / n + 2
            d.rectangle([x - 2, y, x + 30, y + 24], fill=(255, 255, 255, 170))
            d.text((x, y), str(r * n + c + 1), fill=(255, 0, 0, 255), font=FONT)
    return im



class Policy:
    def __init__(self, model="Cloudflare/clef-flash"):
        if model.startswith("Qwen/"):                 # controlled comparison: generative VLM, same records
            from gen_backend import GenVLM
            self.clef = GenVLM(model, max_side=768)
        else:
            self.clef = Clef(model, max_side=768)

    def rec(self, st, ims, qs):
        return {"state": st, "images": ims, "questions": qs}

    def noul(self, st, items, instr):
        return self.clef.noul([(st, ims) for ims in items], instr)

    def cell_dist(self, img, n, st, instr, extra=()):
        q = {"cell": {"type": "choice", "instructions": instr, "criteria": {str(i + 1): f"cell {i + 1}" for i in range(n * n)}}}
        p = self.clef.decide([self.rec(st, [numbered(img, n), *extra], q)])[0]["cell"]
        return [p[str(i + 1)] for i in range(n * n)]

    def pick_cell(self, img, n, st, instr, extra=()):
        k = argmax(self.cell_dist(img, n, st, instr, extra))
        return k // n, k % n

    def refine(self, img, st, instr, extra, r, c, n1, n2):
        W, H = img.size
        cw, ch = W / n1, H / n1
        x0, y0 = max(0, (c - 0.5) * cw), max(0, (r - 0.5) * ch)
        x1, y1 = min(W, (c + 1.5) * cw), min(H, (r + 1.5) * ch)
        r2, c2 = self.pick_cell(img.crop((int(x0), int(y0), int(x1), int(y1))), n2, st, instr, extra)
        return x0 + (c2 + 0.5) * (x1 - x0) / n2, y0 + (r2 + 0.5) * (y1 - y0) / n2

    def locate(self, img, st, instr, extra=(), n1=6, n2=4):
        """coarse-to-fine point in natural image pixels"""
        r, c = self.pick_cell(img, n1, st, instr, extra)
        return self.refine(img, st, instr, extra, r, c, n1, n2)

    def locate_all(self, img, st, instr, items, n1=6, n2=4, top=8):
        """several different items located jointly: each item's distribution over the coarse cells, then the cells
        (no two the same) with the highest product, then each item refined around its cell"""
        P = [self.cell_dist(img, n1, st, instr, (it,)) for it in items]
        cands = [sorted(range(n1 * n1), key=lambda j: -p[j])[:top] for p in P]
        best, ks = -1e18, tuple(c[0] for c in cands)
        if len(items) <= 6:
            for kk in itertools.product(*cands):
                if len(set(kk)) == len(kk):
                    s = sum(math.log(P[i][k] + 1e-9) for i, k in enumerate(kk))
                    if s > best:
                        best, ks = s, kk
        return [self.refine(img, st, instr, (it,), k // n1, k % n1, n1, n2) for it, k in zip(items, ks)]

    # ---------------------------------------------------------------- per-type decisions
    def __call__(self, obs):
        t = obs["type"]
        st = {"puzzle_instruction": obs["prompt"]}
        f = getattr(self, "do_" + t)
        return f(obs, st)

    def do_verify(self, obs, st):
        return argmax(self.noul(st, [[obs["reference"], c] for c in obs["candidates"]], VERIFY))

    do_Image_Matching = do_Object_Match = do_Path_Finder = do_Connect_icon = do_verify

    def do_Rotation_Match(self, obs, st):
        r = self.clef.decide([self.rec(st, [obs["reference"]], {"d": choice("In which direction does the hand or arrow point?", DIRS)})])[0]["d"]
        cs = [c["d"] for c in self.clef.decide([self.rec(st, [o], {"d": choice("In which direction does the top of the character's head point?", DIRS)}) for o in obs["candidates"]])]
        n = len(cs)
        if n != 8:
            return argmax(match(r, cs))
        base = [sum(math.log(cs[i][DIRS[(d0 + i) % 8]] + 1e-6) for i in range(8)) for d0 in range(8)]
        m = max(base)
        pb = [math.exp(x - m) for x in base]
        z = sum(pb)
        pb = [x / z for x in pb]
        return argmax([sum(pb[d0] * r[DIRS[(d0 + i) % 8]] for d0 in range(8)) for i in range(8)])

    def do_Coordinates(self, obs, st):
        qr = {"row": choice("Which row letter is shown in the image?", LETTERS), "col": choice("Which column number is shown in the image?", NUM(100))}
        qc = {"row": choice("Which row letter labels the row where the character is sitting?", LETTERS),
              "col": choice("Which column number labels the column where the character is sitting?", NUM(100))}
        r = self.clef.decide([self.rec(st, [obs["reference"]], qr)])[0]
        cs = self.clef.decide([self.rec(st, [o], qc) for o in obs["candidates"]])
        a = match(r["row"], [c["row"] for c in cs])
        b = match(r["col"], [c["col"] for c in cs])
        return argmax([x * y for x, y in zip(a, b)])

    def do_Dart_Count(self, obs, st):
        r = self.clef.decide([self.rec(st, [obs["reference"]], {"n": choice("What number is shown in the image?", NUM(100))})])[0]["n"]
        cs = self.clef.decide([self.rec(st, [o], {"n": choice("Add up the numbers written in the board sections where the darts landed. What is the total?", NUM(100))}) for o in obs["candidates"]])
        return argmax(match(r, [c["n"] for c in cs]))

    def select(self, p, single=False):
        if single:
            return argmax(p)
        s = [i for i, x in enumerate(p) if x > 0.5]
        return s or [argmax(p)]

    def do_Image_Recognition(self, obs, st):
        p = self.noul(st, [[c] for c in obs["cells"]], "This image is one tile of a 3x3 selection puzzle. Should this tile be selected according to the puzzle instruction?")
        return self.select(p)

    def do_Unusual_Detection(self, obs, st):
        p = self.noul(st, [[c] for c in obs["cells"]], "This picture is one cell of the puzzle grid. Should this cell be selected according to the puzzle instruction?")
        return self.select(p)

    def do_Bingo(self, obs, st):
        tiles = obs["cells"]
        n = len(tiles)
        pairs = list(itertools.combinations(range(n), 2))
        p = self.noul({"task": "Compare two tiles of a picture grid."}, [[tiles[i], tiles[j]] for i, j in pairs],
                      "Do image 1 and image 2 show the same picture (the same object or animal in the same style)?")
        same = [[1.0] * n for _ in range(n)]
        for (i, j), v in zip(pairs, p):
            same[i][j] = same[j][i] = v
        best, arg = -1, [0, 1]
        for i, j in pairs:
            pos = list(range(n))
            pos[i], pos[j] = pos[j], pos[i]
            for ln in LINES3:
                a_, b_, c_ = (pos[k] for k in ln)
                s = same[a_][b_] * same[b_][c_] * same[a_][c_]
                if s > best:
                    best, arg = s, [i, j]
        return arg

    def target(self, prompt):
        p = prompt.strip().rstrip(".")
        for k in ("Select all squares with ", "Select all images with ", "Pick a ", "Pick an "):
            if p.startswith(k):
                return p[len(k):]
        return p

    def do_Patch_Select(self, obs, st):
        img = obs["image"]
        k = int(round(math.sqrt(obs["n"])))
        bs = grid_boxes(*img.size, k, k)
        tgt = self.target(obs["prompt"])
        items = []
        for b in bs:
            im = img.copy()
            ImageDraw.Draw(im).rectangle(b, outline=(255, 0, 0), width=max(3, img.size[0] // 120))
            items.append([im, img.crop(b)])
        p = self.noul(st, items, f"Image 1 is the whole picture with one square outlined in red; image 2 is that square. Does the red square contain any part of a {tgt}?")
        return self.select(p)

    def do_Select_Animal(self, obs, st):
        img = obs["image"]
        rows, cols = obs.get("grid", (2, 3))
        tgt = self.target(obs["prompt"])
        p = self.noul(st, [[img.crop(b)] for b in grid_boxes(*img.size, rows, cols)], f"Does this picture show a {tgt}?")
        return argmax(p)

    def to_display(self, obs, x, y):
        W, H = obs["image"].size
        w, h = obs["display"]
        return x * w / W, y * h / H

    GEN = "The image is divided into numbered red cells. {p} Which numbered cell contains the exact point to click?"

    def do_click(self, obs, st):
        x, y = self.locate(obs["image"], st, self.GEN.format(p=obs["prompt"]))
        return self.to_display(obs, x, y)

    MARK_Q = {
        "Place_Dot": "A red dot with a white ring has been placed on the picture. Is the red dot exactly at the end of the car's path (the far end of the green route, away from the car)?",
        "Pick_Area": "A red dot with a white ring has been placed on the picture. Is the red dot inside the largest area enclosed by the dotted lines?",
        "Geometry_Click": "A red dot with a white ring has been placed on the picture. The instruction is: '{p}'. Is the red dot on the correct target?",
        "Slide_Puzzle": "A puzzle piece has been placed on the picture. Does the piece sit exactly in its gap, so the picture looks complete and seamless?",
    }

    def render_verify(self, img, st, q, render, coarse=10, fine=5, topk=3):
        """verify the rendered result of every lattice point, then a finer lattice around the best coarse points"""
        W, H = img.size
        pts = [((i + .5) * W / coarse, (j + .5) * H / coarse) for j in range(coarse) for i in range(coarse)]
        p = self.clef.noul([(st, [render(x, y)]) for x, y in pts], q)
        fine_pts = []
        for k in sorted(range(len(p)), key=lambda k: -p[k])[:topk]:
            cx, cy = pts[k]
            for j in range(fine):
                for i in range(fine):
                    fine_pts.append((cx + (i - (fine - 1) / 2) * W / coarse / fine, cy + (j - (fine - 1) / 2) * H / coarse / fine))
        pf = self.clef.noul([(st, [render(x, y)]) for x, y in fine_pts], q)
        return fine_pts[argmax(pf)]

    def do_mark_click(self, obs, st):
        img = obs["image"]

        def render(x, y):
            im = img.copy()
            r = max(6, img.size[0] // 60)
            d = ImageDraw.Draw(im)
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(255, 255, 255))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
            return im
        x, y = self.render_verify(img, st, self.MARK_Q[obs["type"]].format(p=obs["prompt"]), render)
        return self.to_display(obs, x, y)

    do_Geometry_Click = do_Pick_Area = do_Place_Dot = do_mark_click

    def do_Slide_Puzzle_render(self, obs, st):
        img, comp = obs["image"], obs["component"]
        if comp.mode != "RGBA":
            comp = comp.convert("RGBA")
        W = img.size[0]
        cw, cd = obs.get("component_display", (None, None))
        w = int(W * cw / obs["display"][0]) if cw else int(0.12 * W)   # piece size as shown on the page
        h = int(w * comp.size[1] / comp.size[0])
        c = comp.resize((w, h))

        def render(x, y):
            im = img.copy()
            im.paste(c, (int(x - w / 2), int(y - h / 2)), c)
            return im
        x, y = self.render_verify(img, st, self.MARK_Q["Slide_Puzzle"], render)
        return self.to_display(obs, x, y)

    def do_Slide_Puzzle(self, obs, st):
        x, y = self.locate(obs["image"], st, "The image is divided into numbered red cells. Image 2 is a puzzle piece. "
                           "Which numbered cell contains the centre of the gap in image 1 where the piece fits?", extra=(obs["component"],), n2=6)
        return self.to_display(obs, x, y)

    def do_Misleading_Click(self, obs, st):
        x, y = self.locate(obs["image"], st, "The image is divided into numbered cells. Which numbered cell is a safe place to click, well away from the item labelled 'Don't click me'?")
        return self.to_display(obs, x, y)

    def do_Click_Order(self, obs, st):
        order = obs["order"]
        q = {"k": choice("How many separate icons are shown in this reference strip?", NUM(9))}
        k = int(max((p := self.clef.decide([self.rec(st, [order], q)])[0]["k"]), key=p.get))
        k = min(max(k, 2), 6)
        W, H = order.size
        icons = [order.crop((i * W // k, 0, (i + 1) * W // k, H)) for i in range(k)]
        pts = []
        for ic in icons:
            x, y = self.locate(obs["image"], st, "The image is divided into numbered red cells. Image 2 shows one icon. "
                               "Which numbered cell contains that icon in image 1?", extra=(ic,))
            pts.append(self.to_display(obs, x, y))
        return pts

    def do_Dice_Count(self, obs, st):
        p = self.clef.decide([self.rec(st, [obs["image"]], {"n": choice("What is the sum of the numbers shown on top of all the dice in the image?", NUM(151))})])[0]["n"]
        return int(max(p, key=p.get))


def make_policy(name):
    return Policy({"clef-flash": "Cloudflare/clef-flash", "clef": "Cloudflare/clef"}.get(name, name))
