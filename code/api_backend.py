"""API-model backend with the same interface as clef.Clef (decide / noul): the skill library's records (state, images,
typed questions) go to a frontier model through an OpenAI-compatible endpoint. Each option's probability comes from
the first-token log-probabilities when the endpoint returns them, otherwise from a probability per option that the
model writes as JSON. Replies are cached on disk by request hash, so reruns cost nothing."""
import os, io, json, time, base64, hashlib, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.expanduser(os.environ.get("API_KEY_FILE", "~/.api_key"))).read().strip()
BASE = os.environ["API_BASE"]
CACHE = os.path.expanduser(os.environ.get("API_CACHE", "~/captcha_jev/results/api_cache.jsonl"))
BUDGET = float(os.environ.get("API_BUDGET_USD", "5"))       # hard stop for this process
PRICE = tuple(map(float, os.environ.get("API_PRICE", "2,10").split(",")))   # USD per 1M input, output tokens


class APIVLM:
    def __init__(self, model, max_side=768, workers=8):
        self.model, self.max_side, self.workers = model, max_side, workers
        self.lock = threading.Lock()
        self.spent = 0.0
        self.cache = {}
        if os.path.exists(CACHE):
            for l in open(CACHE):
                r = json.loads(l)
                self.cache[r["h"]] = r["p"]

    def _b64(self, im):
        w, h = im.size
        s = self.max_side / max(w, h)
        if s < 1:
            im = im.resize((max(28, round(w * s)), max(28, round(h * s))))
        buf = io.BytesIO(); im.convert("RGB").save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _options(self, q):
        if q["type"] == "noul":
            return ["true", "false"], ["yes", "no"]
        crit = q["criteria"]
        if isinstance(crit, dict):
            return list(crit), list(crit)
        return [str(i) for i in range(len(crit))], [str(i) for i in range(len(crit))]

    def _post(self, body):
        if self.spent >= BUDGET:
            raise RuntimeError(f"API budget {BUDGET} USD reached")
        req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        for k in range(5):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=300))
                u = r.get("usage", {})
                with self.lock:
                    self.spent += (u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]) / 1e6
                return r
            except Exception:
                time.sleep(5 * (k + 1))
        return None

    def _one(self, rec, qid, q):
        ids, words = self._options(q)
        state = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"])
        ims = [self._b64(i) for i in rec.get("images", [])]
        ask = q.get("instructions", qid)
        h = hashlib.md5(json.dumps([self.model, state, ask, words, ims]).encode()).hexdigest()
        if h in self.cache:
            return dict(zip(ids, self.cache[h]))
        text = (f"Context: {state}\n{ask}\nGive the probability of each answer as a JSON object with exactly the keys "
                f"{json.dumps(words)} and values summing to 1. Output only the JSON.")
        content = [{"type": "image_url", "image_url": {"url": u}} for u in ims] + [{"type": "text", "text": text}]
        r = self._post({"model": self.model, "messages": [{"role": "user", "content": content}]})
        p = [1 / len(words)] * len(words)
        try:
            txt = r["choices"][0]["message"]["content"]
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            v = [max(float(d.get(w, 0)), 0.0) for w in words]
            if sum(v) > 0:
                p = [x / sum(v) for x in v]
        except Exception:
            pass
        with self.lock:
            self.cache[h] = p
            with open(CACHE, "a") as f:
                f.write(json.dumps({"h": h, "p": p}) + "\n")
        return dict(zip(ids, p))

    def decide(self, records):
        jobs = [(i, qid, q) for i, r in enumerate(records) for qid, q in r["questions"].items()]
        out = [{} for _ in records]
        with ThreadPoolExecutor(self.workers) as ex:
            for (i, qid, _), p in zip(jobs, ex.map(lambda j: self._one(records[j[0]], j[1], j[2]), jobs)):
                out[i][qid] = p
        return out

    def noul(self, items, instruction):
        recs = [{"state": st, "images": ims, "questions": {"q": {"type": "noul", "instructions": instruction}}} for st, ims in items]
        return [a["q"]["true"] for a in self.decide(recs)]
