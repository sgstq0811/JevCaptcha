"""The OCW skill library on COGNITION (static tasks: an instruction and its images, answers as data). The agent is not
told the task type: Clef sees the instruction, the first image and what the task provides (number of images, a tile
grid if the task gives one) and picks a skill; the skills are the OCW ones with the page actions replaced by answers.

  choose : image 1 is the reference, images 2..n the states            -> SkillAgent.choose            -> index
  rotate : image 2 is an object to turn until it matches image 1        -> 8 turned renders, choose     -> angle
  select : tiles (the task's images, or the grid cells of one picture)  -> tile Noul, one/all choice    -> indices
  swap   : swap two tiles so three identical ones line up               -> pairwise same + line search  -> [i, j]
  point  : one point in the picture                                     -> SkillAgent.point_in          -> x, y
  order  : click items of image 2 in the order of the strip in image 1  -> count, locate each           -> points
  type   : a number                                                     -> decade, then digit           -> value
The router also sees the answer rules stated in the task prompt; point and swap need one picture. A point on a tiled
picture answers with the tile under it; a set of tiles is a selection. The answer is written in the format the chosen skill produces; scoring is the artifact's evaluate_pass1."""
import os, re, sys, json, argparse, itertools
from PIL import Image, ImageOps, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from ocw_skills import SkillAgent, ABL, pick_all
from ocw_policy import LINES3, grid_boxes, argmax, choice, NUM

SKILLS = {
    "choose": "image 1 is a reference; choose the one image among the others that matches it as the instruction asks",
    "rotate": "turn the object in image 2 until it matches the reference in image 1",
    "select": "a grid of tiles or squares; select the tiles that fit the instruction",
    "swap": "a grid of tiles; swap two tiles so that identical tiles line up",
    "point": "click one point on a picture",
    "order": "click several items in the order given by a reference",
    "type": "type a number answer",
}
DATA = os.environ.get("COG_DATA", os.path.expanduser("~/captcha_jev/cognition/cognition-usenixsec26-artifact/captcha_data"))
GTS = {}


WORDS = "zero one two three four five six seven eight nine".split()


def count_in(instr):
    """the number an instruction asks tiles to have ('tiles where the shape has 5 holes'), or None"""
    m = re.search(r"\b(?:has|have|having|with|contains?|showing)\s+(?:exactly\s+)?(\d|zero|one|two|three|four|five|six|seven|eight|nine)\b(?!-)", instr.lower())
    return None if not m else int(m.group(1)) if m.group(1).isdigit() else WORDS.index(m.group(1))


def grid_of(t, pid, gt):
    """the tile grid the task provides (its ground-truth record's grid_size), if any"""
    if "grid_size" in (gt or {}):
        return tuple(gt["grid_size"])
    d = {"Connect_Icon": "Connect_icon"}.get(t, t)
    if d not in GTS:
        GTS[d] = json.load(open(f"{DATA}/{d}/ground_truth.json"))
    g = GTS[d].get(pid, {}).get("grid_size")
    return tuple(g) if g else None


def load(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


CUTS = (" Indexing Rules", "\nIndexing Rules", " The 3x3 grid cells", "\nReturn", " Return JSON")


def base_prompt(p):
    for cut in CUTS:
        if cut in p:
            p = p.split(cut)[0]
    return p.strip()


def answer_rules(p):
    """the part of the task prompt that states the answer format"""
    return p[len(base_prompt(p)):].strip()[:400] or "none given"


class StaticSkills(SkillAgent):
    def solve(self, prompt, paths, grid, rules):
        st = {"instruction": prompt}
        ims = [load(p) for p in paths]
        aff = {"images": len(ims), "tile_grid": f"{grid[0]} x {grid[1]}" if grid else "none", "answer_rules": rules}
        opts = {k: v for k, v in SKILLS.items() if len(ims) == 1 or k not in ("point", "swap") or "router" in ABL}   # a point or a swap needs one picture
        opts = {k: v for k, v in opts.items() if len(ims) > 1 or k not in ("choose", "rotate") or "router" in ABL}   # states need a second picture
        sk = REPLAY.get(self.key) or self.ask({**st, "task_provides": aff}, ims[:1], "You are solving the puzzle in the image. Which kind of interaction does it need?", opts)
        if sk == "point" and grid and "set" in rules and "setsel" not in ABL:    # a set of tiles is a selection
            sk = "select"
        ans, log = getattr(self, "st_" + sk)(st, ims, paths, grid)
        return sk, ans, log

    def tiles_of(self, ims, grid):
        if grid:
            im = ims[0]
            boxes = grid_boxes(*im.size, *grid)
            return im, boxes, [im.crop(b) for b in boxes]
        return None, None, ims

    def st_choose(self, st, ims, paths, grid):
        attr, k, p = self.choose(st, ims[0], ims[0], ims[1:])
        return {"answer_type": "classify", "index": k}, f"{attr} {k}"

    def st_rotate(self, st, ims, paths, grid):
        obj = Image.open(paths[-1]).convert("RGBA")
        cands = []
        for ang in range(0, 360, 45):                  # the object turned clockwise in 45-degree steps
            r = obj.rotate(-ang, expand=True)
            bg = Image.new("RGB", r.size, (255, 255, 255))
            bg.paste(r, mask=r.split()[3])
            cands.append(bg)
        attr, k, p = self.choose(st, ims[0], ims[0], cands)
        return {"answer_type": "rotation", "angle": 45.0 * k}, f"{attr} {k}"

    def st_select(self, st, ims, paths, grid):
        _, _, tiles = self.tiles_of(ims, grid)
        n = count_in(st["instruction"]) if "count" not in ABL else None
        if n is not None:                               # v18: tiles picked by a count the instruction names (rule B against the read number)
            q = choice(f"The instruction is: '{st['instruction']}'. How many of the things the instruction counts does this picture show?", NUM(10))
            p = [d["n"].get(str(n), 0.0) for d in self.clef.decide([self.pol.rec(st, [t], {"n": q}) for t in tiles])]
            sel = pick_all(p)
            return {"answer_type": "multi_select", "indices": sorted(sel)}, f"count {n} {sel}"
        p = self.tile_p(st, tiles)
        one = self.ask(st, ims[:1], "Does the instruction ask for exactly one tile or for all tiles that fit?", {"one": "exactly one", "all": "all that fit"})
        sel = [argmax(p)] if one == "one" else pick_all(p)
        return {"answer_type": "multi_select", "indices": sorted(sel)}, f"{one} {sel}"

    def st_swap(self, st, ims, paths, grid):
        _, _, cr = self.tiles_of(ims, grid or (3, 3))
        n = len(cr)
        pairs = list(itertools.combinations(range(n), 2))
        p = self.noul({"task": "Compare two tiles of a picture grid."}, [[cr[i], cr[j]] for i, j in pairs],
                      "Do image 1 and image 2 show the same picture (the same object or animal in the same style)?")
        if "swapver" in ABL and n == 9:                # variant: one question per rendered swap
            return {"correct": list(self.swap_render_pick(cr, pairs))}, "swap (rendered)"
        same = [[1.0] * n for _ in range(n)]
        for (i, j), v in zip(pairs, p):
            same[i][j] = same[j][i] = v
        best, arg = -1, [0, 1]
        for i, j in pairs:
            pos = list(range(n))
            pos[i], pos[j] = pos[j], pos[i]
            for m, ln in enumerate(LINES3):
                if max(ln) >= n:
                    continue
                a, b, c = (pos[k] for k in ln)
                s = same[a][b] * same[b][c] * same[a][c] * (1 if m < 6 or "axis" in ABL else .5)   # a row or column before a diagonal
                if s > best:
                    best, arg = s, [i, j]
        return {"correct": arg}, f"swap {arg}"

    def st_point(self, st, ims, paths, grid):
        x, y = self.point_in(st, ims[-1])
        if grid:                                       # a point on a tiled picture selects the tile under it
            r, c = grid
            W, H = ims[-1].size
            k = min(r - 1, int(y * r / H)) * c + min(c - 1, int(x * c / W))
            return {"answer_type": "multi_select", "indices": [k]}, f"point -> tile {k}"
        return {"answer_type": "single_point", "point": {"x": x, "y": y}}, f"point ({x:.0f}, {y:.0f})"

    def st_order(self, st, ims, paths, grid):
        if len(ims) < 2:
            return self.st_point(st, ims, paths, grid)
        order, img = ims[0], ims[1]
        k = int(self.ask(st, [order], "How many separate items are shown in this reference?", {str(i): str(i) for i in range(2, 9)}))
        W, H = order.size
        horiz = W >= H
        items = [order.crop((i * W // k, 0, (i + 1) * W // k, H) if horiz else (0, i * H // k, W, (i + 1) * H // k)) for i in range(k)]
        q = "The image is divided into numbered red cells. Image 2 shows one item. Which numbered cell contains that item in image 1?"
        pts = [self.pol.locate(img, st, q, extra=(it,)) for it in items] if "assign" in ABL else self.pol.locate_all(img, st, q, items)
        return {"answer_type": "multi_point", "points": [{"x": x, "y": y} for x, y in pts]}, f"order {len(pts)}"

    def st_type(self, st, ims, paths, grid):
        v = self.number(st, ims[0])
        return {"answer_type": "number", "value": int(v)}, f"type {v}"


# ROUTE_FROM: a log file whose skills replace the router (crossed router x scorer study)
REPLAY = {r["key"]: r["skill"] for r in map(json.loads, open(os.environ["ROUTE_FROM"]))} if os.environ.get("ROUTE_FROM") else {}
StaticSkills.key = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.expanduser("~/captcha_jev/results/cog_tasks.json"))
    ap.add_argument("--split", default="full")
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="all")
    a = ap.parse_args()
    ag = StaticSkills(a.model, True)
    tasks = [d for d in json.load(open(a.tasks))[a.split] if a.types == "all" or d["type"] in a.types.split(",")]
    preds = json.load(open(a.out)) if os.path.exists(a.out) else {}
    logf = open(a.out.replace(".json", "_log.jsonl"), "a")
    for d in tasks:
        if d["key"] in preds:
            continue
        ag.key = d["key"]
        try:
            sk, ans, log = ag.solve(base_prompt(d["prompt"]), d["images"], grid_of(d["type"], d["pid"], d["gt"]), answer_rules(d["prompt"]))
        except Exception as e:
            sk, ans, log = None, None, f"ERR {type(e).__name__} {e}"
            print("ERR", d["key"], log, flush=True)
        preds[d["key"]] = ans
        logf.write(json.dumps({"key": d["key"], "skill": sk, "log": log}) + "\n")
        logf.flush()
        json.dump(preds, open(a.out, "w"))
    print("done", len(preds))
