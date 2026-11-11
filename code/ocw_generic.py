"""Generic Jev agent for Open CaptchaWorld. The agent is not told the puzzle type and has no per-type code.
Each step it sees a screenshot of the puzzle area with the page's interactive elements outlined and numbered
(found from the DOM by element kind only: buttons, inputs, pointer/grab cursors, images), the instruction text and
its own action history, and Clef chooses the next action with one typed decision:
  click k | click a point inside k (numbered-cell choice, coarse then fine) | type a number into k |
  drag k to a point | press and hold k
With --tools the agent may also choose three generic tools, defined by element kind only:
  explore k : click k repeatedly until the puzzle area repeats a previous look; a Noul 'does the puzzle now satisfy
              the instruction?' scores every look; return to the best one
  check group g : elements of identical size form a group; a Noul per element 'should it be selected?'; click those
              above 0.5 (the best one if none)
  try points in k : draw a dot at lattice points of image k (10x10, then 5x5 around the best 3); Noul 'is the dot on
              the right place?'; click the best point
  drag-watch k : hold draggable k and move it over an 8x8 lattice of the puzzle area (then 5x5 around the best 3);
              Noul on each live preview (whole puzzle + zoom on the dragged element); release at the best
  swap group g : paste-swap every pair of the group's crops into the screenshot; Noul per picture; click the best pair
Explore scores each look on the whole puzzle plus a zoom on the pixels that changed between looks. Holding releases
once the puzzle area has changed and then settled; numbers are chosen decade first, then digit (v2).
The episode ends when the page submits an answer (official /api/check_answer verdict) or after MAX_STEPS."""
import io, os, sys, json, time, argparse
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from ocw_agent import URL, GT_ROOT, TYPES
from ocw_policy import Policy, NUM, choice

MAX_STEPS = 20
FONT = ImageFont.load_default(size=15)
FIND = """() => {
  const root = document.querySelector('#puzzle-container'); const R = root.getBoundingClientRect(); const out = [];
  for (const e of root.querySelectorAll('*')) {
    const cs = getComputedStyle(e);
    if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) continue;
    const b = e.getBoundingClientRect();
    if (b.width < 8 || b.height < 8 || b.bottom < R.top || b.top > R.bottom) continue;
    const tag = e.tagName.toLowerCase();
    const big = (tag === 'img' || tag === 'canvas') && b.width >= 60 && b.height >= 60;
    const act = ['button', 'input', 'select', 'textarea', 'a'].includes(tag) || ['pointer', 'grab', 'move'].includes(cs.cursor)
                || e.onclick || e.getAttribute('role') === 'button' || e.draggable;
    if (!act && !big) continue;
    out.push({tag, x: b.x, y: b.y, w: b.width, h: b.height, cursor: cs.cursor, big,
              input: tag === 'input' || tag === 'textarea', drag: e.draggable || ['grab', 'move'].includes(cs.cursor),
              text: ((e.innerText || e.value || e.placeholder || '') + '').trim().replace(/\\s+/g, ' ').slice(0, 24)});
  }
  return out;
}"""


def dedupe(els):
    """one entry per on-screen box (nested wrappers with the same box collapse), at most 40"""
    keep = []
    for e in els:
        same = [k for k in keep if abs(k["x"] - e["x"]) < 3 and abs(k["y"] - e["y"]) < 3 and abs(k["w"] - e["w"]) < 3 and abs(k["h"] - e["h"]) < 3]
        if same:
            k = same[0]
            for f in ("big", "input", "drag"):
                k[f] = k[f] or e[f]
            k["text"] = k["text"] or e["text"]
            continue
        keep.append(e)
    return keep[:40]


def groups(els):
    """sets of >= 3 non-input elements with the same size (cells of a grid, tiles of a strip)"""
    gs = {}
    for i, e in enumerate(els, 1):
        if e["input"] or e["big"] and e["w"] > 300:
            continue
        gs.setdefault((round(e["w"] / 4), round(e["h"] / 4)), []).append(i)
    return [v for v in gs.values() if len(v) >= 3]


SOLVED = "Does the puzzle in the image now satisfy the instruction (ready to submit)?"
SOLVED_Z = "Image 1 is the whole puzzle, image 2 zooms into the part that the action changes. Does the puzzle now satisfy the instruction (ready to submit)?"


def changed_box(looks, pad=.15):
    """bounding box of the pixels that differ between the looks, padded"""
    import numpy as np
    a = np.stack([np.asarray(im.convert("L").resize(looks[0].size), dtype=np.int16) for im in looks])
    m = (np.abs(a - a[0]).max(0) > 30)
    H, W = m.shape
    if not m.any():
        return (0, 0, W, H)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = pad * (x1 - x0) + 10, pad * (y1 - y0) + 10
    return (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py)))


class GenericAgent:
    def __init__(self, model, tools=False):
        self.pol = Policy(model)
        self.clef = self.pol.clef
        self.tools = tools

    def noul(self, st, ims, q):
        return self.clef.noul([(st, i) for i in ims], q)

    def explore(self, pg, st, e):
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        looks, keys = [], []
        for _ in range(16):
            im, _ = self.shot(pg)
            key = im.resize((96, 96)).convert("L").tobytes()
            if any(max(abs(a - b) for a, b in zip(key, k)) < 24 for k in keys):
                break
            looks.append(im); keys.append(key)
            pg.mouse.click(cx, cy)
            pg.wait_for_timeout(250)
        if len(looks) < 2:
            return "no visible change"
        looks = [im.resize(looks[0].size) for im in looks]
        z = changed_box(looks)
        if len(looks) <= 12:                           # the looks side by side: which one satisfies the instruction?
            q = {"k": {"type": "choice", "criteria": {f"look {i + 1}": f"image {i + 2}" for i in range(len(looks))},
                       "instructions": "Image 1 is the whole puzzle. Images 2.. zoom into the part that the action changes, one per state it can take "
                                       "(look 1 = image 2, look 2 = image 3, ...). In which look does the puzzle satisfy the instruction?"}}
            c = self.clef.decide([self.pol.rec(st, [looks[0]] + [im.crop(z) for im in looks], q)])[0]["k"]
            p = [c[f"look {i + 1}"] for i in range(len(looks))]
        else:
            p = self.noul(st, [[im, im.crop(z)] for im in looks], SOLVED_Z)
        best = max(range(len(p)), key=p.__getitem__)
        n = len(looks)
        for _ in range(best):                          # we are back at look 0 after the cycle
            pg.mouse.click(cx, cy)
            pg.wait_for_timeout(250)
        return f"{n} looks, kept look {best + 1} (P={p[best]:.2f})"

    def check_group(self, pg, st, els, g):
        im, box = self.shot(pg)
        crops = [im.crop((round(els[i - 1]["x"] - box["x"]), round(els[i - 1]["y"] - box["y"]),
                          round(els[i - 1]["x"] - box["x"] + els[i - 1]["w"]), round(els[i - 1]["y"] - box["y"] + els[i - 1]["h"]))) for i in g]
        p = self.noul(st, [[im, c] for c in crops], "Image 2 is one element of the puzzle in image 1. Should this element be selected or clicked according to the instruction?")
        sel = [i for i, v in zip(g, p) if v > 0.5] or [g[max(range(len(p)), key=p.__getitem__)]]
        for i in sel:
            e = els[i - 1]
            pg.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
            pg.wait_for_timeout(150)
        return f"selected elements {sel}"

    def try_points(self, pg, st, e, im, box):
        crop = im.crop((round(e["x"] - box["x"]), round(e["y"] - box["y"]), round(e["x"] - box["x"] + e["w"]), round(e["y"] - box["y"] + e["h"])))

        def render(x, y):
            o = crop.copy()
            r = max(5, crop.size[0] // 60)
            d = ImageDraw.Draw(o)
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(255, 255, 255))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
            return o
        x, y = self.pol.render_verify(crop, st, "A red dot with a white ring marks a candidate point. Is the red dot exactly where the instruction says to click?", render)
        pg.mouse.click(e["x"] + x * e["w"] / crop.size[0], e["y"] + y * e["h"] / crop.size[1])
        return f"clicked ({x / crop.size[0]:.2f}, {y / crop.size[1]:.2f})"

    def drops(self, pg, st, e, area=None):
        """hold element e and move it over a lattice of the puzzle area (or of `area`); score each preview; release at the best"""
        _, box = self.shot(pg)
        ab, n = (box, 8) if not area else ({"x": area["x"], "y": area["y"], "width": area["w"], "height": area["h"]}, 10)
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        pg.mouse.move(cx, cy)
        pg.mouse.down()

        def look(pts):
            ims = []
            for x, y in pts:
                pg.mouse.move(x, y, steps=4)
                pg.wait_for_timeout(60)
                im, _ = self.shot(pg)
                r = max(e["w"], e["h"])
                ims.append([im, im.crop((int(max(0, x - box["x"] - 1.5 * r)), int(max(0, y - box["y"] - 1.5 * r)),
                                         int(min(im.size[0], x - box["x"] + 1.5 * r)), int(min(im.size[1], y - box["y"] + 1.5 * r))))])
            return self.noul(st, ims, "Image 1 is the whole puzzle while an element is being dragged, image 2 zooms into the dragged element. "
                                      "If it is released here, does the puzzle satisfy the instruction?")
        W, H = ab["width"], ab["height"]
        pts = [(ab["x"] + (i + .5) * W / n, ab["y"] + (j + .5) * H / n) for j in range(n) for i in range(n)]
        p = look(pts)
        fine = []
        for k in sorted(range(len(p)), key=lambda k: -p[k])[:3]:
            for j in range(5):
                for i in range(5):
                    fine.append((pts[k][0] + (i - 2) * W / n / 5, pts[k][1] + (j - 2) * H / n / 5))
        pf = look(fine)
        b = max(range(len(pf)), key=pf.__getitem__)
        pg.mouse.move(*fine[b], steps=4)
        pg.wait_for_timeout(100)
        pg.mouse.up()
        return f"dropped at ({(fine[b][0] - ab['x']) / W:.2f}, {(fine[b][1] - ab['y']) / H:.2f}) (P={pf[b]:.2f})"

    def swap(self, pg, st, els, g):
        """imagine swapping every pair of same-size elements (paste the crops), score each picture, click the best pair"""
        im, box = self.shot(pg)
        bx = [(round(els[i - 1]["x"] - box["x"]), round(els[i - 1]["y"] - box["y"]),
               round(els[i - 1]["x"] - box["x"] + els[i - 1]["w"]), round(els[i - 1]["y"] - box["y"] + els[i - 1]["h"])) for i in g]
        cr = [im.crop(b) for b in bx]
        pairs = [(a, b) for a in range(len(g)) for b in range(a + 1, len(g))]
        ren = []
        for a, b in pairs:
            o = im.copy()
            o.paste(cr[b].resize(cr[a].size), bx[a][:2])
            o.paste(cr[a].resize(cr[b].size), bx[b][:2])
            ren.append([o])
        p = self.noul(st, ren, "The picture shows the puzzle after swapping two of its elements. Does the puzzle now satisfy the instruction?")
        a, b = pairs[max(range(len(p)), key=p.__getitem__)]
        for i in (g[a], g[b]):
            e = els[i - 1]
            pg.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
            pg.wait_for_timeout(200)
        return f"swapped elements {g[a]} and {g[b]} (P={max(p):.2f})"

    def hold(self, pg, e):
        """press e and keep holding while the puzzle area keeps changing; release once it has changed and then settled"""
        pg.mouse.move(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
        pg.mouse.down()
        t0, prev, moved, still = time.time(), None, False, 0
        while time.time() - t0 < 10:
            pg.wait_for_timeout(200)
            key = self.shot(pg)[0].resize((96, 96)).convert("L").tobytes()
            if prev is not None:
                if max(abs(a - b) for a, b in zip(key, prev)) >= 24:
                    moved, still = True, 0
                else:
                    still += 1
            prev = key
            if moved and still >= 2:
                break
        pg.mouse.up()
        return f"held {time.time() - t0:.1f}s"

    def number(self, st, im):
        """decade first, then the digit"""
        q = {"d": choice("What number should be typed to answer the instruction? Choose its range.", [f"{10 * i}-{10 * i + 9}" for i in range(20)])}
        d = self.clef.decide([self.pol.rec(st, [im], q)])[0]["d"]
        lo = int(max(d, key=d.get).split("-")[0])
        q = {"n": choice("What number should be typed to answer the instruction?", [str(lo + i) for i in range(10)])}
        v = self.clef.decide([self.pol.rec(st, [im], q)])[0]["n"]
        return max(v, key=v.get)

    def shot(self, pg):
        box = pg.locator("#puzzle-container").bounding_box()
        im = Image.open(io.BytesIO(pg.screenshot(clip=box))).convert("RGB")
        return im, box

    def marked(self, im, box, els):
        o = im.copy()
        d = ImageDraw.Draw(o)
        for i, e in enumerate(els, 1):
            x, y = e["x"] - box["x"], e["y"] - box["y"]
            d.rectangle([x, y, x + e["w"], y + e["h"]], outline=(255, 0, 0), width=2)
            d.rectangle([x, y, x + 9 * len(str(i)) + 6, y + 18], fill=(255, 0, 0))
            d.text((x + 3, y + 1), str(i), fill=(255, 255, 255), font=FONT)
        return o

    def options(self, els):
        ops = {}
        for i, e in enumerate(els, 1):
            lab = f"element {i}" + (f" ('{e['text']}')" if e["text"] else "")
            if e["input"]:
                ops[f"type:{i}"] = f"type a number into {lab}"
                continue
            ops[f"click:{i}"] = f"click {lab}"
            if e["big"]:
                ops[f"point:{i}"] = f"click a specific point inside {lab}"
            if e["drag"] or e["big"]:
                ops[f"drag:{i}"] = f"drag {lab} to another place"
            if e["tag"] == "button" or e["cursor"] == "pointer":
                ops[f"hold:{i}"] = f"press and hold {lab}"
        return ops

    def point(self, im, st, instr, extra=()):
        return self.pol.locate(im, st, "The image is divided into numbered red cells. " + instr + " Which numbered cell contains the exact point?", extra)

    def step(self, pg, prompt, hist):
        if not hist:
            self.keys = []
        im, box = self.shot(pg)
        els = dedupe(pg.evaluate(FIND))
        ops = self.options(els)
        gs = groups(els) if self.tools else []
        if self.tools:
            for i, e in enumerate(els, 1):
                if not e["input"] and not e["big"]:
                    ops[f"explore:{i}"] = f"explore: click element {i} repeatedly through all its states and keep the best one"
                if e["big"]:
                    ops[f"try:{i}"] = f"try candidate points inside element {i} and click the best one"
                if e["drag"]:
                    ops[f"drops:{i}"] = f"drag element {i} over the puzzle, watch each position and drop it at the best one"
            for j, g in enumerate(gs):
                ops[f"group:{j}"] = f"check each of the similar elements {g[0]}-{g[-1]} and select the right ones"
                if 4 <= len(g) <= 16:
                    ops[f"swap:{j}"] = f"imagine swapping each pair of the similar elements {g[0]}-{g[-1]} and swap the best pair"
        if len(self.keys) >= 3 and len(set(self.keys[-3:])) == 1:     # no 4th repeat of an action
            ops.pop(self.keys[-1], None)
        if len(self.keys) >= 4 and len({k.split(":")[1] for k in self.keys[-4:]}) == 1:   # 4 actions on one element: try another
            el = self.keys[-1].split(":")[1]
            ops = {k: v for k, v in ops.items() if k.split(":")[1] != el or k.startswith(("group", "swap"))}
        if self.keys and self.keys[-1].split(":")[0] in ("explore", "try", "drops", "group", "swap"):   # tools are deterministic
            ops.pop(self.keys[-1], None)
        st = {"instruction": prompt, "actions_so_far": hist or "none", "steps_left": MAX_STEPS - len(hist)}
        q = {"a": {"type": "choice", "instructions": "You are solving the puzzle in the image (interactive elements are outlined and numbered in red). "
                   "Which action should be taken next to complete the instruction and submit the answer?", "criteria": ops}}
        p = self.clef.decide([self.pol.rec(st, [self.marked(im, box, els)], q)])[0]["a"]
        a = max(p, key=p.get)
        self.keys.append(a)
        kind, k = a.split(":")
        if kind == "group":
            return ops[a] + ": " + self.check_group(pg, st, els, gs[int(k)])
        if kind == "swap":
            return ops[a] + ": " + self.swap(pg, st, els, gs[int(k)])
        e = els[int(k) - 1]
        if kind == "drops":
            return ops[a] + ": " + self.drops(pg, st, e)
        if kind == "explore":
            return ops[a] + ": " + self.explore(pg, st, e)
        if kind == "try":
            return ops[a] + ": " + self.try_points(pg, st, e, im, box)
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        desc = ops[a]
        if kind == "click":
            pg.mouse.click(cx, cy)
        elif kind == "point":
            crop = im.crop((round(e["x"] - box["x"]), round(e["y"] - box["y"]), round(e["x"] - box["x"] + e["w"]), round(e["y"] - box["y"] + e["h"])))
            x, y = self.point(crop, st, f"Instruction: {prompt} Where inside this image should be clicked?")
            pg.mouse.click(e["x"] + x * e["w"] / crop.size[0], e["y"] + y * e["h"] / crop.size[1])
            desc += f" at ({x / crop.size[0]:.2f}, {y / crop.size[1]:.2f})"
        elif kind == "drag":
            x, y = self.point(im, st, f"Instruction: {prompt} Element {k} (outlined) will be dragged. Where should it be dropped?", )
            tx, ty = box["x"] + x, box["y"] + y
            pg.mouse.move(cx, cy)
            pg.mouse.down()
            for j in range(1, 26):
                pg.mouse.move(cx + (tx - cx) * j / 25, cy + (ty - cy) * j / 25)
                pg.wait_for_timeout(15)
            pg.mouse.up()
            desc += f" to ({x / im.size[0]:.2f}, {y / im.size[1]:.2f})"
        elif kind == "hold":
            desc += ": " + self.hold(pg, e)
        elif kind == "type":
            v = self.number(st, im)
            pg.mouse.click(cx, cy)
            pg.keyboard.press("Control+A")
            pg.keyboard.type(v)
            desc += f": {v}"
        pg.wait_for_timeout(300)
        return desc


def run(agent, types, out, limit=0):
    from playwright.sync_api import sync_playwright   # only the browser runs need it
    with sync_playwright() as p:
        b = p.chromium.launch()
        done = {}
        if os.path.exists(out):
            for l in open(out):
                r = json.loads(l)
                done[r["type"]] = done.get(r["type"], 0) + 1
        for t in types:                                  # the type only selects which puzzles to serve
            n = len(json.load(open(f"{GT_ROOT}/{t}/ground_truth.json")))
            n = n if not limit else min(limit, n)
            if done.get(t, 0) >= n:
                continue
            if done.get(t):                              # restart an unfinished type from scratch
                rows = [l for l in open(out) if json.loads(l)["type"] != t]
                open(out, "w").writelines(rows)
            ctx = b.new_context(viewport={"width": 1280, "height": 1800})
            pg = ctx.new_page()

            served = []

            def handler(route, t=t, pg=pg):
                r = pg.request.get(f"{URL}/api/get_puzzle?debug_type={t}")
                served.append(r.json()["puzzle_id"])
                route.fulfill(response=r)
            pg.route("**/api/get_puzzle**", lambda route: handler(route))
            verdict = []

            def judge(route, pg=pg):                      # pass the answer to the official checker, keep its verdict
                r = pg.request.fetch(route.request)
                verdict.append(r.json())
                route.fulfill(response=r)
            pg.route("**/api/check_answer**", lambda route: judge(route))
            agent.verdict = verdict
            ok = 0
            for i in range(n):
                pg.goto("about:blank")
                served.clear()
                pg.goto(URL)
                pg.wait_for_timeout(1500)
                pid = served[-1] if served else None
                pg.evaluate("() => { const d = document.querySelector('#debug-indicator'); if (d) d.remove(); }")
                prompt = pg.inner_text("#puzzle-prompt")
                verdict.clear()
                log = {"type": t, "puzzle_id": pid, "steps": []}
                try:
                    while not verdict and len(log["steps"]) < MAX_STEPS:
                        s = agent.step(pg, prompt, [f"{j + 1}. {s}" for j, s in enumerate(log["steps"])])
                        if s is None:                    # the agent has finished its episode
                            break
                        log["steps"].append(s)
                    for _ in range(50):
                        if verdict:
                            break
                        pg.wait_for_timeout(200)
                    v = verdict[0] if verdict else {}
                    log["correct"] = bool(v.get("correct"))
                    log["server"] = v
                except Exception as e:
                    log["correct"] = False
                    log["error"] = f"{type(e).__name__}: {e}"[:300]
                ok += log["correct"]
                with open(out, "a") as f:
                    f.write(json.dumps(log) + "\n")
            print(f"{t}: {ok}/{n}", flush=True)
            pg.unroute_all(behavior="ignoreErrors")
            pg.goto("about:blank")
            ctx.close()
        b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(GenericAgent(a.model, a.tools), a.types, a.out, a.limit)
