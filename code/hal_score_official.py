"""Score states with the official Halligan Flask server (in-process test client; verifier code unchanged)."""
import os, sys, json, collections
import hal_data as H
rows = [json.loads(l) for l in open(os.path.abspath(sys.argv[1]))]
B = os.path.dirname(H.ROOT)
os.chdir(B)
sys.path.insert(0, B)
from server import app

cl = app.test_client()
res = collections.defaultdict(list)
for r in rows:
    t, i, s = r["type"], r["id"], r["state"]
    ok = False
    if s is not None:
        c = H.load(t)[i - 1]
        body = {"id": i, "state": s}
        if t == "hcaptcha":
            body["challenge_type"] = "binary" if c["subtype"] in ("hcaptcha_type_2", "hcaptcha_type_3") else "area"
        resp = cl.post(H.TYPES[t][1].format(id=i), json=body)
        ok = resp.status_code == 200 and bool(resp.json.get("solved"))
    res[t].append(ok)
tot = n = 0
for t in H.TYPES:
    if t in res:
        print(f"{t:36s} {sum(res[t]):3d}/{len(res[t])}")
        tot += sum(res[t]); n += len(res[t])
print(f"TOTAL {tot}/{n} = {100 * tot / max(n, 1):.1f}%")
