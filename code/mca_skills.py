"""The frozen skill library on the MCA-Bench official test split (held out: never used while building the library).
The adapter only describes each page as Halligan pages are described (a picture with a Submit button; a 3 x 3 grid
of tiles; a picture with a text box) and writes the skill's output in MCA's answer format; routing, skills and rules
are those of hal_skills.py. Scoring is the official MCA rule (mca_tasks.score)."""
import os, sys, json, time, io, base64, argparse, traceback
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import mca_tasks as M
from hal_skills import HalSkills, View, SKILLS, ABL, edit
from ocw_policy import grid_boxes


def b64(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def view(name, path):
    p = Image.open(path).convert("RGB")
    kind = M.TASKS[name][2]
    if kind == "label_set_sorted":                     # the page shows the picture cut into a 3 x 3 grid of tiles
        bx = grid_boxes(*p.size, 3, 3)
        return View({"buttons": ["Verify"], "grids_of_same_size_tiles": [9]}, p, tiles=(p, bx, [p.crop(b) for b in bx]),
                    fmt_select=lambda sel, pr: ",".join(map(str, sorted(sel))))
    if kind in ("exact", "exact_insensitive"):          # a picture of text and a text box
        return View({"buttons": ["Submit"], "pictures": 1, "text_boxes": 1}, p, text={"instruction_image": [b64(path)]})
    return View({"buttons": ["Submit"], "pictures": 1}, p, picture=p, fmt_point=lambda x, y: f"({round(x)},{round(y)})")


class MCASkills(HalSkills):
    def solve(self, name, path, prompt):
        v = view(name, path)
        st = {"instruction": prompt}
        a = v.aff
        can = {"cycle": len(a.get("buttons", [])) > 2 or "sliders" in a, "select": "grids_of_same_size_tiles" in a,
               "swap": "grids_of_same_size_tiles" in a, "point": "pictures" in a, "order": "pictures" in a,
               "drag": "draggable" in a or "sliders" in a, "type": "text_boxes" in a, "hold": bool(a.get("buttons"))}
        opts = {k: d for k, d in SKILLS.items() if can.get(k, True) or "router" in ABL}
        sk = self.ask({**st, "page_elements": a}, [v.scene], "You are solving the puzzle in the image. Which kind of interaction does it need?", opts)
        f = getattr(self, "h_" + sk, None)
        out = f(st, v) if f else None
        if sk == "type" and out and name == "word":    # MCA writes the two words with a space: split at the best dictionary boundary
            out = split_words(out, self.pol.lex) if hasattr(self.pol, "lex") else out
        return sk, out


def split_words(s, lex):
    s = s.replace(" ", "")
    words = {w for ws in lex.values() for w in ws}
    best = min(range(1, len(s)), key=lambda k: (s[:k] not in words) + (s[k:] not in words) + abs(2 * k - len(s)) / (10 * len(s)))
    return s[:best] + " " + s[best:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--tasks", default=",".join(M.TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    done = {(r["task"], r["i"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    ag = MCASkills(a.model)
    with open(a.out, "a") as f:
        for t in a.tasks.split(","):
            _, test = M.load_split(t)
            for i, it in enumerate(test[:a.limit]):
                if (t, i) in done:
                    continue
                path, prompt, gold = M.parse_item(it)
                t0 = time.time()
                try:
                    sk, pred = ag.solve(t, path, prompt)
                except Exception:
                    traceback.print_exc()
                    sk, pred = None, None
                ok = bool(pred) and M.score(t, it, str(pred))
                f.write(json.dumps({"task": t, "i": i, "skill": sk, "pred": pred, "gold": gold, "ok": ok, "sec": round(time.time() - t0, 2)}) + "\n")
                f.flush()
            print(t, "done", flush=True)
