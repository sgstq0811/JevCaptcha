"""The OCW skill library on Halligan (26 commercial CAPTCHA types, official challenges and Flask verifier). The agent
is not told the type: each challenge is shown as its page would be (pictures, tile grids, text box, draggable pieces,
controls), Clef picks one skill from the OCW library, the skill answers, and the answer is written in the state
format of that page's submit endpoint. A skill that the page cannot express gives no answer.

Page views (the interface of each endpoint, not its solution):
  pictures + text box                 text CAPTCHAs (mtcaptcha, botdetect, yandex text)
  picture to click                    tencent, amazon, hcaptcha area types
  tiles                               recaptcha, hcaptcha binary types, arkose multichoice sprites (2 x 3)
  reference + pages behind arrows     arkose paged
  picture + slider through states     baidu (turns the picture), yandex kaleidoscope (applies the next swap)
  board + draggable piece(s)          geetest slide (horizontal track), lemin (three pieces, 9.8 px grid)
  picture + reference icons           geetest icon (click in the order of the icons)
  grid of icons, swap two             geetest iconcrush (3 x 3), gobang (5 x 5)"""
import os, sys, json, time, string, argparse, itertools, traceback
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
import hal_data as H
from hal_data import img
from hal_policy import HalPolicy, rgba, icon_img, lines
from ocw_skills import SkillAgent, SKILLS, fit_map, ABL, pick_all, WHOLE
from ocw_policy import grid_boxes, argmax
from tools import ocr


def edit(a, b):
    """Levenshtein distance"""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def page_text(t, c):
    """the instruction as the page shows it (reCAPTCHA pages wrap the keyword in a sentence)"""
    ins = (c.get("instruction") or PAGE_TEXT.get(t, "")).replace("*", "")
    if t == "recaptchav2":
        ins = f"Select all images with {ins}" if c.get("subtype") == "binary" else f"Select all squares with {ins}. If there are none, click skip"
    return ins


class View:
    def __init__(self, aff, scene, **kw):
        self.aff, self.scene, self.instr = aff, scene, None
        self.__dict__.update(kw)


PAGE_TEXT = {"yandex/text": "Enter the code from the image",            # the instruction the page itself shows when
             "yandex/kaleidoscope": "Move the slider to complete the puzzle",  # the challenge record has none
             "geetest/iconcrush": "Click and swap to line up three identical items in a row",
             "geetest/gobang": "Click and drop to line up five identical items in a row",
             "geetest/icon": "Select in this order:"}


def view(t, c):
    ins = c.get("instruction") or ""
    sub = c.get("subtype")
    if t in ("mtcaptcha", "botdetect", "yandex/text"):
        p = img(c["instruction_image"][0])
        return View({"buttons": ["Submit"], "pictures": 1, "text_boxes": 1}, p, text=c)
    if t in ("tencent", "amazon") or t == "hcaptcha" and sub in ("hcaptcha_type_4", "hcaptcha_type_5"):
        p = img(c["images"][0] if t == "hcaptcha" else c["instruction_image"][0])
        return View({"buttons": ["Submit"], "pictures": 1}, p, picture=p, fmt_point=lambda x, y: [x / p.size[0], y / p.size[1]])
    if t == "hcaptcha" or t == "recaptchav2" and sub == "binary":
        ts = [img(b) for b in c["images"]]
        ex = img(c["instruction_image"][0]) if c.get("instruction_image") else None
        return View({"buttons": ["Verify"], "grids_of_same_size_tiles": [len(ts)]}, ex or ts[0], tiles=(None, None, ts), example=ex,
                    fmt_select=lambda sel, p: [i in sel for i in range(len(ts))])
    if t == "recaptchav2":
        w = img(c["images"][0])
        bx = grid_boxes(*w.size, 4, 4)
        return View({"buttons": ["Verify"], "grids_of_same_size_tiles": [16]}, w, tiles=(w, bx, [w.crop(b) for b in bx]),
                    fmt_select=lambda sel, p: [False] * 4 + [i in sel for i in range(16)])
    if t.startswith("arkose/multichoice"):
        w = img(c["instruction_image"][0])
        bx = grid_boxes(*w.size, 2, 3)
        return View({"grids_of_same_size_tiles": [6]}, w, tiles=(w, bx, [w.crop(b) for b in bx]), fmt_select=lambda sel, p: argmax(p))
    if t.startswith("arkose/paged"):
        r = img(c["instruction_image"][0])
        cs = [img(b) for b in c["images"]]
        h = max(r.size[1], cs[0].size[1])
        pg = Image.new("RGB", (r.size[0] + cs[0].size[0] + 20, h), (255, 255, 255))   # the page: reference left, state right
        pg.paste(r, (0, 0)); pg.paste(cs[0], (r.size[0] + 20, 0))
        return View({"buttons": ["←", "→", "Submit"], "pictures": 2}, r, states=(r, cs), page=pg, fmt_cycle=lambda k: k)
    if t == "baidu":
        p = img(c["instruction_image"][0])
        angs = list(range(0, 360, 10))
        cs = [p.rotate(-a, resample=Image.BICUBIC, fillcolor=(255, 255, 255)) for a in angs]
        return View({"pictures": 1, "sliders": 1}, p, states=(None, cs), fmt_cycle=lambda k: float(angs[k]))
    if t == "yandex/kaleidoscope":
        p = img(c["instruction_image"][0])
        tiles = [p.crop(b) for b in grid_boxes(*p.size, 4, 4)]
        tw, th = tiles[0].size
        state, cs = list(range(16)), []
        for k in range(len(c["swaps"]) + 1):           # the slider applies the page's swaps one after another
            if k:
                a, b = c["swaps"][k - 1]
                state[a], state[b] = state[b], state[a]
            o = Image.new("RGB", (tw * 4, th * 4))
            for pos, q in enumerate(state):
                o.paste(tiles[q], ((pos % 4) * tw, (pos // 4) * th))
            cs.append(o)
        return View({"pictures": 1, "sliders": 1}, p, states=(None, cs), fmt_cycle=lambda k: k)
    if t == "geetest/slide":
        bd = img(c["instruction_image"][0])
        pc = rgba(c["images"][0])
        y = c["height"]
        return View({"buttons": [], "pictures": 1, "draggable": 1}, bd, board=bd, pieces=[np.array(pc)], track_y=y,
                    fmt_drag=lambda k, x, yy: (x - pc.size[0] / 2) / bd.size[0])
    if t == "lemin":
        im = img(c["instruction_image"][0])
        bd = im.crop((0, 0, 400, 400))
        sc = 1180 / 1156.4
        w = 117.6 * sc
        ps = []
        for k in range(c["n_pieces"]):                 # the page's piece tray
            x0, y0 = (396.9 + k * 127.4) * sc, 4.9 * sc
            a = np.array(im.crop((round(x0), round(y0), round(x0 + w), round(y0 + w))))
            ps.append(np.dstack([a, ((a.sum(2) > 30) * 255).astype(np.uint8)]))
        g = 9.8 * sc
        return View({"pictures": 1, "draggable": c["n_pieces"]}, im, board=bd, pieces=ps, track_y=None,
                    fmt_drag=lambda k, x, y: [[k, round(x / g), round(y / g)]])
    if t == "geetest/icon":
        p = img(c["images"][0])
        icons = [icon_img(b) for b in c["instruction_images"]]
        strip = Image.new("RGB", (sum(i.size[0] for i in icons), max(i.size[1] for i in icons)), (255, 255, 255))
        x = 0
        for i in icons:
            strip.paste(i, (x, 0)); x += i.size[0]
        return View({"buttons": ["Submit"], "pictures": 2}, p, picture=p, order=icons, strip=strip,
                    fmt_order=lambda pts: [[x / p.size[0], y / p.size[1]] for x, y in pts])
    if t in ("geetest/iconcrush", "geetest/gobang"):
        grid = c["grid"]
        n = len(grid)
        cells = [(r, q) for r in range(n) for q in range(n)]
        ims = {rc: img(c["images"][grid[rc[0]][rc[1]]]) if c["images"][grid[rc[0]][rc[1]]] else None for rc in cells}
        tw = max(i.size[0] for i in ims.values() if i)
        board = Image.new("RGB", (tw * n, tw * n), (255, 255, 255))
        for (r, q), i in ims.items():
            if i:
                board.paste(i.resize((tw, tw)), (q * tw, r * tw))

        def fmt(a, b):
            g2 = [row[:] for row in grid]
            (r1, c1), (r2, c2) = a, b
            g2[r1][c1], g2[r2][c2] = g2[r2][c2], g2[r1][c1]
            return g2
        return View({"grids_of_same_size_tiles": [n * n]}, board, swapgrid=(n, cells, ims), fmt_swap=fmt)
    raise KeyError(t)


# ROUTE_FROM: a results file whose skills replace the router (crossed router x scorer study)
REPLAY = {(r["type"], r["id"]): r["skill"] for r in map(json.loads, open(os.environ["ROUTE_FROM"]))} if os.environ.get("ROUTE_FROM") else {}


class HalSkills(SkillAgent):
    key = None

    def __init__(self, model):
        self.pol = HalPolicy(model)
        self.clef = self.pol.clef
        self.tools = True

    def solve(self, t, c):
        v = view(t, c)
        st = {"instruction": page_text(t, c)}
        a = v.aff                                       # the skills the page's elements allow
        can = {"cycle": len(a.get("buttons", [])) > 2 or "sliders" in a, "select": "grids_of_same_size_tiles" in a,
               "swap": "grids_of_same_size_tiles" in a, "point": "pictures" in a, "order": "pictures" in a,
               "drag": "draggable" in a or "sliders" in a, "type": "text_boxes" in a, "hold": bool(a.get("buttons"))}
        opts = {k: d for k, d in SKILLS.items() if can.get(k, True) or "router" in ABL}
        sk = REPLAY.get(self.key) or self.ask({**st, "page_elements": a}, [v.scene], "You are solving the puzzle in the image. Which kind of interaction does it need?", opts)
        if sk == "drag" and hasattr(v, "states") and "slider" not in ABL:     # dragging a slider moves the picture through its states
            sk = "cycle"
        f = getattr(self, "h_" + sk, None)
        return sk, (f(st, v) if f else None)

    def h_cycle(self, st, v):
        if not hasattr(v, "states"):
            return None
        r, cs = v.states
        _, k, _ = self.choose(st, v.page if hasattr(v, "page") and "page" not in ABL else v.scene, r, cs)   # what to match: asked on the whole page
        return v.fmt_cycle(k)

    def h_select(self, st, v):
        if not hasattr(v, "tiles"):
            return None
        _, _, ts = v.tiles
        p = self.tile_p(st, ts, getattr(v, "example", None))
        one = self.ask(st, [v.scene], "Does the instruction ask for exactly one tile or for all tiles that fit?", {"one": "exactly one", "all": "all that fit"})
        sel = [argmax(p)] if one == "one" else pick_all(p)
        return v.fmt_select(sel, p)

    def h_swap(self, st, v):
        if not hasattr(v, "swapgrid"):
            return None
        n, cells, ims = v.swapgrid                     # a line of identical icons across the whole grid, any direction
        live = [rc for rc in cells if ims[rc] is not None]
        pairs = list(itertools.combinations(live, 2))
        p = self.noul({"task": "Compare two tiles of a picture grid."}, [[ims[a], ims[b]] for a, b in pairs],
                      "Do image 1 and image 2 show the same picture (the same object or animal in the same style)?")
        same = {}
        for (a, b), x in zip(pairs, p):
            same[a, b] = same[b, a] = x
        for a in cells:
            same[a, a] = 1.0
        best, arg = -1, None
        for a, b in itertools.combinations(cells, 2):
            pos = {rc: rc for rc in cells}
            pos[a], pos[b] = b, a
            for j, ln in enumerate(lines(n, True)):
                src = [pos[x] for x in ln]
                if any(ims[s] is None for s in src):
                    continue
                s = 1.0 if j < 2 * n or "axis" in ABL else 0.5   # a row or column before a diagonal
                for u, w in itertools.combinations(src, 2):
                    s *= same.get((u, w), 0.0)
                if s > best:
                    best, arg = s, (a, b)
        return v.fmt_swap(*arg) if arg else None

    def h_point(self, st, v):
        if not hasattr(v, "picture"):
            return None
        return v.fmt_point(*self.point_in(st, v.picture)) if hasattr(v, "fmt_point") else None

    def h_order(self, st, v):
        if not hasattr(v, "order"):
            return None
        q = "The image is divided into numbered red cells. Image 2 shows one item. Which numbered cell contains that item in image 1?"
        pts = [self.pol.locate(v.picture, st, q, extra=(it,)) for it in v.order] if "assign" in ABL else self.pol.locate_all(v.picture, st, q, v.order)
        return v.fmt_order(pts)

    def h_drag(self, st, v):
        if not hasattr(v, "pieces"):
            return None
        bd = np.array(v.board)
        best, cands = None, []
        for k, pc in enumerate(v.pieces):              # each piece's best fit on the board; the piece that fits best
            fm = fit_map(bd, pc, pc.shape[1] / 1.0, pc.shape[0] / 1.0)
            if fm is None:
                continue
            r, w, h = fm
            r = r.copy()
            q = 6
            r[:q], r[-q:], r[:, :q], r[:, -q:] = -1, -1, -1, -1
            if v.track_y is not None:                  # a horizontal track: only the piece's own row
                y = min(max(0, v.track_y + (pc.shape[0] - h) // 2), r.shape[0] - 1)
                row = r[y].copy()
                row[:q] = -1
                x = int(np.argmax(row))
                cand = (row[x], k, x + w / 2, y + h / 2)
            else:
                y, x = np.unravel_index(int(np.argmax(r)), r.shape)
                cand = (r[y, x], k, x + w / 2, y + h / 2)
            cands.append((cand, w, h))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            return None
        if len(cands) > 1 and "pieces" not in ABL:     # several pieces: each placed at its best fit, judged as a whole picture
            ims = []
            for (_, k, x, y), w, h in cands:
                o = v.board.convert("RGB")
                p = Image.fromarray(v.pieces[k]).resize((w, h))
                o.paste(p, (round(x - w / 2), round(y - h / 2)), p)
                ims.append([o])
            best = cands[argmax(self.noul(st, ims, WHOLE))][0]
        _, k, x, y = best
        return v.fmt_drag(k, x, y)

    def h_type(self, st, v):
        if not hasattr(v, "text"):
            return None
        cs = self.ask(st, [v.scene], "Which characters does the text in this image use?",
                      {"upper": "capital letters and digits", "lower": "small letters and digits", "words": "small letters only (words)"})
        tool = "tool" not in ABL
        if cs == "words":                              # English words: read word by word against a dictionary
            w = self.pol.do_text(v.text, st)
            if not tool:
                return w
            return self.verify_words(st, img(v.text["instruction_image"][0]), [w] + self.ocr_words(img(v.text["instruction_image"][0])))
        chars = {"upper": string.ascii_uppercase + string.digits, "lower": string.ascii_lowercase + string.digits}[cs]
        extra = [r.replace(" ", "") for r in ocr(img(v.text["instruction_image"][0]))] if tool else []
        return self.pol.read_text(v.text, st, list(chars), 20, extra=[r for r in extra if r])

    def ocr_words(self, im):
        """OCR readings as word sequences, each word also snapped to its nearest dictionary words (edit distance <= 2)"""
        lex = self.pol.lex                             # loaded by do_text
        out = []
        for r in ocr(im):
            ws = [w.lower() for w in r.split() if w.isalpha()]
            if not ws:
                continue
            out.append("".join(ws))
            alts = [sorted((x for n in range(len(w) - 2, len(w) + 3) for x in lex.get(n, [])), key=lambda x: edit(x, w))[:3] if len(ws) <= 3 else [w] for w in ws]
            for combo in itertools.product(*alts):
                if all(edit(a, w) <= 2 for a, w in zip(combo, ws)):
                    out.append("".join(combo))
        return list(dict.fromkeys(out))

    def verify_words(self, st, im, cands):
        cands = list(dict.fromkeys(c for c in cands if c))
        v = self.clef.decide([self.pol.rec(st, [im], {"v": {"type": "noul", "instructions": f"Does the text in this image read exactly '{s}' (ignoring spaces)?"}}) for s in cands])
        return cands[argmax([d["v"]["true"] for d in v])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--types", default="all")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="results file; run only the challenges it lists")
    a = ap.parse_args()
    types = list(H.TYPES) if a.types == "all" else a.types.split(",")
    done = {(r["type"], r["id"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    only = {(r["type"], r["id"]) for r in map(json.loads, open(a.only))} if a.only else None
    ag = HalSkills(a.model)
    with open(a.out, "a") as f:
        for t in types:
            for i, c in enumerate(H.load(t)[:a.n], 1):
                if (t, i) in done or (only and (t, i) not in only):
                    continue
                t0 = time.time()
                ag.key = (t, i)
                try:
                    sk, s = ag.solve(t, c)
                except Exception:
                    traceback.print_exc()
                    sk, s = None, None
                f.write(json.dumps({"type": t, "id": i, "skill": sk, "state": s, "sec": round(time.time() - t0, 2)}) + "\n")
                f.flush()
            print(t, "done", flush=True)
