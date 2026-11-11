"""Direct-prompting baseline with an API model (OpenAI-compatible chat endpoint) on MCA-Bench and COGNITION: the same
prompts as mca_direct.py / cog_direct.py (official instruction + answer format; official system prompt + schema), images
as data URLs. For MCA point tasks the image size is stated so that the model answers in pixels. Key read from
$API_KEY_FILE; endpoint $API_BASE."""
import os, sys, io, json, base64, time, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--bench", choices=["mca", "cog"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tasks", help="cog: tasks json; mca: comma list of tasks")
ap.add_argument("--split", default="rest")
ap.add_argument("--limit", type=int, default=0, help="per task (mca) / total (cog)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
KEY = open(os.path.expanduser(os.environ.get("API_KEY_FILE", "~/.api_key"))).read().strip()
BASE = os.environ["API_BASE"]   # an OpenAI-compatible endpoint


def b64(im):
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(msgs):
    body = json.dumps({"model": a.model, "messages": msgs}).encode()
    for k in range(6):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", body,
                                         {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            return r["choices"][0]["message"].get("content") or "", r.get("usage", {})
        except Exception as e:
            err = str(e)[:200]
            time.sleep(10 * (k + 1))
    return None, {"error": err}


done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
jobs = []
if a.bench == "mca":
    import mca_tasks as M
    for t in (a.tasks.split(",") if a.tasks else list(M.TASKS)):
        _, test = M.load_split(t)
        for i, it in enumerate(test[:a.limit or None]):
            if f"{t}/{i}" not in done:
                jobs.append((f"{t}/{i}", t, it))
else:
    from cog_direct_common import extract_json, SYSTEM
    SCHEMAS = json.load(open(os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/results/cog_schemas.json")))
    for d in json.load(open(a.tasks))[a.split][:a.limit or None]:
        if d["key"] not in done:
            jobs.append((d["key"], None, d))


def load(p, side=1024):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    s = side / max(im.size)
    return im.resize((round(im.size[0] * s), round(im.size[1] * s))) if s < 1 else im


def run(job):
    key, t, it = job
    t0 = time.time()
    if a.bench == "mca":
        path, prompt, gold = M.parse_item(it)
        im = Image.open(path).convert("RGB")
        kind = M.TASKS[t][2]
        size = f" The image is {im.size[0]}x{im.size[1]} pixels." if kind in ("first_region", "all_regions") else ""
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(im)}},
                                             {"type": "text", "text": prompt + size + M.FORMAT[kind]}]}]
        out, use = chat(msgs)
        rec = {"key": key, "task": t, "pred": out, "gold": gold, "ok": bool(out is not None and M.score(t, it, out))}
    else:
        text = it["prompt"] + "\n\nSchema:\n" + json.dumps(SCHEMAS[it["type"]], ensure_ascii=False)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [*({"type": "image_url", "image_url": {"url": b64(load(p))}} for p in it["images"]),
                                             {"type": "text", "text": text}]}]
        out, use = chat(msgs)
        rec = {"key": key, "reply": out, "parsed": extract_json(out) if out else None}
    rec.update(usage=use, sec=round(time.time() - t0, 1))
    return rec


with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
    for rec in ex.map(run, jobs):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
print("done", len(jobs))
