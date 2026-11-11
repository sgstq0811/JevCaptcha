"""Generic perception tools for the skill library (Florence-2-large, open weights): they only propose candidates
(text readings); every decision is still a Clef question. Outputs are cached on disk by image hash so
that reruns are deterministic and need no model when cached.

  ocr(img, k)   -> candidate readings of the text in the picture (beam search, both line orders joined)
  objects(img)  -> boxes of the things in the picture (object detection and region proposals joined)"""
import os, io, json, hashlib, threading
from PIL import Image

CACHE = os.path.expanduser(os.environ.get("TOOL_CACHE", os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/results/tool_cache"))
MODEL = "florence-community/Florence-2-large"
_m = {}
_lock = threading.Lock()


def _key(img, task):
    b = io.BytesIO()
    img.convert("RGB").save(b, format="PNG")
    return hashlib.sha1(b.getvalue() + task.encode()).hexdigest()


def _cached(img, task, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/{_key(img, task)}.json"
    if os.path.exists(p):
        return json.load(open(p))
    v = fn(img.convert("RGB"))
    json.dump(v, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return v


def _load():
    with _lock:
        if not _m:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _m["p"] = AutoProcessor.from_pretrained(MODEL)
            _m["m"] = Florence2ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            _m["dev"] = dev
    return _m["p"], _m["m"], _m["dev"]


def _gen(img, task, beams, n):
    import torch
    p, m, dev = _load()
    x = p(text=task, images=img, return_tensors="pt").to(dev, m.dtype)
    with torch.inference_mode():
        o = m.generate(**x, max_new_tokens=512, num_beams=beams, num_return_sequences=n, do_sample=False)
    return p, p.batch_decode(o, skip_special_tokens=False)


def ocr(img, k=5):
    def run(im):
        if im.size[0] < 400:                            # small text images are read upscaled
            im = im.resize((400, round(400 * im.size[1] / im.size[0])), Image.LANCZOS)
        p, outs = _gen(im, "<OCR>", max(k, 5), max(k, 5))
        out = []
        for t in outs:
            s = p.post_process_generation(t, task="<OCR>", image_size=im.size)["<OCR>"]
            ls = [l.strip() for l in s.split("\n") if l.strip()]
            for v in (" ".join(ls), " ".join(ls[::-1])):     # lines may come in either order
                if v and v not in out:
                    out.append(v)
        return out
    return _cached(img, f"ocr{k}", run)[:2 * k]



def objects(img):
    def run(im):
        out = []
        for task in ("<OD>", "<REGION_PROPOSAL>"):
            p, outs = _gen(im, task, 3, 1)
            out += [[float(v) for v in b] for b in p.post_process_generation(outs[0], task=task, image_size=im.size)[task]["bboxes"]]
        return out
    return _cached(img, "objects", run)
