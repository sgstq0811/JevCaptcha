"""Direct-prompting baseline on COGNITION with a local generative VLM: the official system prompt, task prompt, schema
and images go to the model, which writes its answer; the official extract_json turns the reply into the parsed answer that
cog_tasks.py score grades. No skills, no enumeration (the setting of the COGNITION paper's model rows)."""
import os, re, json, argparse, torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText


def extract_json(text):
    """verbatim from the artifact's cognition/run_eval.py"""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I|re.M)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, (dict, list)) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/results/cog_tasks.json"))
ap.add_argument("--split", default="main")
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--out", required=True)
ap.add_argument("--schemas", default=os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/results/cog_schemas.json"))
a = ap.parse_args()
SCHEMAS = json.load(open(a.schemas))   # the artifact's build_json_schema per family
SYSTEM = ("You are a vision agent. Output ONLY a JSON object that strictly matches the given schema. "
          "If the schema contains optional field 'reasoning', you MUST fill it following the detailed cognitive tracing format provided.")

proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()


def load(p, side=1024):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    s = side / max(im.size)
    return im.resize((round(im.size[0] * s), round(im.size[1] * s))) if s < 1 else im


preds = json.load(open(a.out)) if os.path.exists(a.out) else {}
raw = open(a.out.replace(".json", "_raw.jsonl"), "a")
for d in json.load(open(a.tasks))[a.split]:
    if d["key"] in preds:
        continue
    ims = [load(p) for p in d["images"]]
    text = d["prompt"] + "\n\nSchema:\n" + json.dumps(SCHEMAS[d["type"]], ensure_ascii=False)   # as the artifact's chat providers
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [*({"type": "image"} for _ in ims), {"type": "text", "text": text}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    x = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        y = model.generate(**x, max_new_tokens=a.max_new, do_sample=False)
    out = proc.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
    preds[d["key"]] = extract_json(out)
    raw.write(json.dumps({"key": d["key"], "reply": out}) + "\n"); raw.flush()
    json.dump(preds, open(a.out, "w"))
print("done", len(preds))
