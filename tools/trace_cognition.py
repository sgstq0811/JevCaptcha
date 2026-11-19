"""Re-run the frozen library on chosen COGNITION challenges and record every closed question, the probabilities
Clef-flash returned and the candidates the skills built (the step-by-step cases of the supplementary material).
usage: python tools/trace_cognition.py <out dir> <key>,<key>,...   e.g. Dart_Count/dart_puzzle_10.json"""
import os, sys, json
os.environ["ABL"] = "count"                      # the frozen v17 library (no count-select)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cog_skills as C

OUT, KEYS = sys.argv[1], sys.argv[2].split(",")
ROOT = os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev"))
tasks = {d["key"]: d for d in json.load(open(ROOT + "/results/cog_tasks.json"))["full"]}
logged = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cog_v17_full.json")))


class Rec(C.StaticSkills):
    """the frozen library, recording each typed question and its returned probabilities"""
    def noul(self, st, groups, q):
        p = super().noul(st, groups, q)
        self.trace.append({"type": "NOUL", "question": q, "n_records": len(groups), "p_true": p})
        return p

    def read(self, st, ims, q):
        p = super().read(st, ims, q)
        self.trace.append({"type": "CHOICE-read", "question": str(q), "n_records": len(ims), "dists": p})
        return p

    def ask(self, st, ims, instr, opts):
        a = super().ask(st, ims, instr, opts)
        self.trace.append({"type": "CHOICE-ask", "question": instr, "options": opts, "answer": a})
        return a

    def tiles_of(self, ims, grid):
        im, boxes, tiles = super().tiles_of(ims, grid)
        for i, t in enumerate(tiles):
            t.save(f"{self.dir}/tile_{i}.png")
        if boxes:
            self.trace.append({"type": "enumerate-tiles", "boxes": [list(b) for b in boxes]})
        return im, boxes, tiles

    def choose(self, st, scene, r, cands):
        attr, k, p = super().choose(st, scene, r, cands)
        self.trace.append({"type": "aggregate", "attribute": attr, "scores": p, "chosen": k})
        for i, c in enumerate(cands):
            c.save(f"{self.dir}/candidate_{i}.png")
        return attr, k, p


ag = Rec("Cloudflare/clef-flash", True)
for key in KEYS:
    d = tasks[key]
    ag.key, ag.trace = key, []
    ag.dir = os.path.join(OUT, key.replace("/", "__").replace(".json", "").replace(".png", ""))
    os.makedirs(ag.dir, exist_ok=True)
    for i, p in enumerate(d["images"]):
        C.load(p).save(f"{ag.dir}/input_{i}.png")
    prompt = C.base_prompt(d["prompt"])
    sk, ans, log = ag.solve(prompt, d["images"], C.grid_of(d["type"], d["pid"], d["gt"]), C.answer_rules(d["prompt"]))
    rec = {"key": key, "instruction": prompt, "images": d["images"], "ground_truth": d["gt"], "skill": sk, "answer": ans,
           "log": log, "logged_answer": logged.get(key), "same_as_logged": ans == logged.get(key), "trace": ag.trace}
    json.dump(rec, open(f"{ag.dir}/trace.json", "w"), indent=1)
    print(key, sk, ans, "same_as_logged=", rec["same_as_logged"], flush=True)
