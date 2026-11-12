"""Delegation of free-click actions to a frontier model through an OpenAI-compatible endpoint ($API_BASE, key in
$API_KEY_FILE): the picture (and the reference strip for an ordered click) with the page instruction; the model writes
pixel coordinates as JSON. Usage is appended to $DELEGATE_LOG; $API_BUDGET_USD stops the process."""
import os, io, re, json, time, base64, urllib.request

MODEL = os.environ.get("DELEGATE", "")
KEY = open(os.path.expanduser(os.environ.get("API_KEY_FILE", "~/.api_key"))).read().strip() if MODEL else ""
BASE = os.environ.get("API_BASE", "")
LOG = os.path.expanduser(os.environ.get("DELEGATE_LOG", "~/captcha_jev/results/delegate_log.jsonl"))
BUDGET = float(os.environ.get("API_BUDGET_USD", "5"))
PRICE = tuple(map(float, os.environ.get("API_PRICE", "2,10").split(",")))
spent = 0.0


def b64(im):
    buf = io.BytesIO(); im.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(ims, text):
    global spent
    if spent >= BUDGET:
        raise RuntimeError(f"API budget {BUDGET} USD reached")
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": [
        *({"type": "image_url", "image_url": {"url": b64(i)}} for i in ims), {"type": "text", "text": text}]}]}).encode()
    for k in range(6):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", body, {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            u = r.get("usage", {})
            spent += (u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]) / 1e6
            out = r["choices"][0]["message"].get("content") or ""
            with open(LOG, "a") as f:
                f.write(json.dumps({"text": text[:200], "reply": out, "usage": u, "spent": round(spent, 4)}) + "\n")
            return out
        except Exception:
            time.sleep(10 * (k + 1))
    return ""


def nums(txt):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", txt)]


def point(img, instr):
    W, H = img.size
    out = ask([img], f"{instr}\nThe image is {W}x{H} pixels. Where should one click to solve this? "
                     'Answer only with JSON {"x": <pixels>, "y": <pixels>}.')
    v = nums(out[out.find("{"):] if "{" in out else out)
    if len(v) < 2:
        return None
    return min(max(v[0], 0), W - 1), min(max(v[1], 0), H - 1)


def order(img, ref, instr):
    W, H = img.size
    out = ask([img, ref], f"{instr}\nImage 1 is the picture to click ({W}x{H} pixels); image 2 is the reference that gives the order. "
                          'Answer only with a JSON list of the points to click in order, [{"x": <pixels>, "y": <pixels>}, ...].')
    v = nums(out[out.find("["):] if "[" in out else out)
    pts = [(min(max(v[i], 0), W - 1), min(max(v[i + 1], 0), H - 1)) for i in range(0, len(v) - 1, 2)]
    return pts or None
