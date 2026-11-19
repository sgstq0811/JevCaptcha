"""Re-run the frozen v17 Halligan library on chosen challenges and record every Clef-flash call (question, number of
records, returned probabilities) and the images the skills build (the step-by-step cases of the supplementary
material); each recorded state can be compared with results/hal_v17_all.jsonl.
usage: python tools/trace_halligan.py <out dir> type:id,type:id,...   e.g. geetest/gobang:1,baidu:1"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import numpy as np
from PIL import Image
import hal_data as H
import hal_skills as S

OUT, KEYS = sys.argv[1], [k.rsplit(":", 1) for k in sys.argv[2].split(",")]


def short(d):
    """a returned distribution, its top entries only"""
    if isinstance(d, dict):
        return {k: (short(v) if isinstance(v, dict) else v) for k, v in sorted(d.items(), key=lambda kv: -kv[1] if isinstance(kv[1], float) else 0)[:12]}
    return d


class Rec(S.HalSkills):
    def noul(self, st, groups, q):
        p = super().noul(st, groups, q)
        k = len(self.trace)
        if len(groups) <= 12:
            for i, g in enumerate(groups):
                for j, im in enumerate(g):
                    im.save(f"{self.dir}/noul{k}_{i}_{j}.png")
        self.trace.append({"type": "NOUL", "question": q, "n": len(groups), "p_true": list(map(float, p)), "img_prefix": f"noul{k}"})
        return p

    def choose(self, st, scene, r, cands):
        attr, k, p = super().choose(st, scene, r, cands)
        self.trace.append({"type": "aggregate", "attribute": attr, "scores": list(map(float, p)), "chosen": int(k)})
        for i, c in enumerate(cands):
            c.save(f"{self.dir}/state_{i}.png")
        return attr, k, p


ag = Rec("Cloudflare/clef-flash")
dec = ag.clef.decide


def decide(recs):
    out = dec(recs)
    qs = [{n: q.get("instructions") for n, q in r["questions"].items()} for r in recs]
    ag.trace.append({"type": "decide", "n": len(recs), "questions": qs, "answers": [{n: short(v) for n, v in o.items()} for o in out]})
    return out


ag.clef.decide = decide
for t, i in KEYS:
    i = int(i)
    c = H.load(t)[i - 1]
    ag.key, ag.trace = (t, i), []
    ag.dir = os.path.join(OUT, f"{t.replace('/', '_')}_{i}")
    os.makedirs(ag.dir, exist_ok=True)
    v = S.view(t, c)
    v.scene.save(f"{ag.dir}/scene.png")
    if hasattr(v, "board"):
        v.board.save(f"{ag.dir}/board.png")
        for k, pc in enumerate(v.pieces):
            Image.fromarray(pc).save(f"{ag.dir}/piece_{k}.png")
    sk, s = ag.solve(t, c)
    json.dump({"type": t, "id": i, "instruction": S.page_text(t, c), "skill": sk, "state": s, "trace": ag.trace},
              open(f"{ag.dir}/trace.json", "w"), indent=1, default=str)
    print(t, i, sk, str(s)[:80], flush=True)
