"""Run the typed-decision policy on Halligan challenges; writes one JSON line per challenge with the state to submit."""
import os, sys, json, time, argparse, traceback
sys.path.insert(0, os.path.dirname(__file__))
import hal_data as H
from hal_policy import HalPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Cloudflare/clef-flash")
ap.add_argument("--types", default="all")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()
types = list(H.TYPES) if a.types == "all" else a.types.split(",")
done = set()
if os.path.exists(a.out):
    done = {(r["type"], r["id"]) for r in map(json.loads, open(a.out))}
pol = HalPolicy(a.model)
with open(a.out, "a") as f:
    for t in types:
        for i, c in enumerate(H.load(t)[:a.n], 1):
            if (t, i) in done:
                continue
            t0 = time.time()
            try:
                s = pol(t, c)
            except Exception:
                traceback.print_exc()
                s = None
            f.write(json.dumps({"type": t, "id": i, "state": s, "sec": round(time.time() - t0, 2), "trace": pol.trace}) + "\n")
            f.flush()
        print(t, "done", flush=True)
