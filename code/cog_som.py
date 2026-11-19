"""Set-of-Mark control: the frozen library's route and candidates, but a generative VLM (Qwen3.5-9B) picks among the
numbered candidates in one answer, replacing Ask + Aggregate. See results/PREREG_cog_som.txt."""
import os, sys, re, json, torch
os.environ["ABL"] = "count"                      # the frozen v17 library (no count-select)
import cog_skills as C
from ocw_policy import grid_boxes
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText

OUT = sys.argv[1]
ROOT = os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev"))
T = json.load(open(ROOT + "/results/cog_tasks.json"))
main = {d["key"] for d in T["main"]}
tasks = [d for d in T["full"] if d["key"] not in main]
route = {}
for l in open(ROOT + "/results/cog_v17_full_log.jsonl"):
    r = json.loads(l); route[r["key"]] = r["skill"]

NAME = "Qwen/Qwen3.5-9B"
proc = AutoProcessor.from_pretrained(NAME)
model = AutoModelForImageTextToText.from_pretrained(NAME, torch_dtype=torch.bfloat16, device_map="cuda").eval()
FONT = ImageFont.load_default(size=28) if hasattr(ImageFont, "load_default") else None


def small(im, m=512):
    im = im.copy(); im.thumbnail((m, m)); return im


@torch.inference_mode()
def ask(images, text):
    images = [small(i) for i in images]
    c = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    t = proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[t], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)


def nums(s):
    return [int(x) for x in re.findall(r"\d+", s)]


def label(im, k):
    im = im.copy(); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 44, 36], fill="white"); d.text((6, 2), str(k), fill="black", font=FONT)
    return im


def cells(im, n=6):
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size
    f = ImageFont.load_default(size=max(12, min(W, H) // 28))
    for j in range(n):
        for i in range(n):
            x0, y0 = i * W / n, j * H / n
            d.rectangle([x0, y0, x0 + W / n, y0 + H / n], outline="red", width=2)
            d.text((x0 + 3, y0 + 2), str(j * n + i + 1), fill="red", font=f, stroke_width=2, stroke_fill="white")
    return im


preds, log = {}, open(OUT.replace(".json", "_log.jsonl"), "w")
for d in tasks:
    key, sk = d["key"], route[d["key"]]
    ins = C.base_prompt(d["prompt"]); paths = d["images"]; ims = [C.load(p) for p in paths]
    grid = C.grid_of(d["type"], d["pid"], d["gt"])
    if sk == "rotate":
        obj = Image.open(paths[-1]).convert("RGBA"); cands = []
        for ang in range(0, 360, 45):
            r = obj.rotate(-ang, expand=True); bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3]); cands.append(bg)
        a = ask([ims[0]] + [label(c, i + 1) for i, c in enumerate(cands)],
                f"Image 1 is the reference. Images 2 to 9 show the object turned into candidates 1 to 8 (the number is in the corner). Instruction: {ins} Which candidate (1-8) is correct? Answer with the number only.")
        n = [x for x in nums(a) if 1 <= x <= 8]
        ans = {"answer_type": "rotation", "angle": 45.0 * (n[0] - 1)} if n else None
    elif sk in ("select", "swap"):
        if grid or sk == "swap":
            g = grid or (3, 3); im = ims[0]; tiles = [im.crop(b) for b in grid_boxes(*im.size, *g)]
        else:
            tiles = ims
        nt = len(tiles)
        if sk == "select":
            a = ask([label(t, i + 1) for i, t in enumerate(tiles)],
                    f"Images 1 to {nt} are the tiles of a puzzle, numbered in reading order. Instruction: {ins} Which tiles should be selected? Answer with the tile numbers separated by commas, nothing else.")
            n = sorted({x - 1 for x in nums(a) if 1 <= x <= nt})
            ans = {"answer_type": "multi_select", "indices": n}
        else:
            a = ask([label(t, i + 1) for i, t in enumerate(tiles)],
                    f"Images 1 to {nt} are the tiles of a 3x3 grid, numbered in reading order. Instruction: {ins} Which two tiles should be swapped? Answer with the two tile numbers separated by a comma, nothing else.")
            n = [x - 1 for x in nums(a) if 1 <= x <= nt][:2]
            ans = {"correct": n} if len(n) == 2 else None
    elif sk == "point":
        im = ims[-1]; W, H = im.size
        a = ask([cells(im)], f"The picture is divided into 36 numbered cells. Instruction: {ins} In which cell (1-36) should one click? Answer with the cell number only.")
        n = [x for x in nums(a) if 1 <= x <= 36]
        if n:
            k = n[0] - 1; x, y = (k % 6 + .5) * W / 6, (k // 6 + .5) * H / 6
            if grid:
                r, c = grid
                ans = {"answer_type": "multi_select", "indices": [min(r - 1, int(y * r / H)) * c + min(c - 1, int(x * c / W))]}
            else:
                ans = {"answer_type": "single_point", "point": {"x": x, "y": y}}
        else:
            ans = None
    else:
        a, ans = "skill not covered", None
    preds[key] = ans
    log.write(json.dumps({"key": key, "skill": sk, "reply": a, "ans": ans}) + "\n"); log.flush()
    print(key, sk, repr(a[:40]), ans, flush=True)
json.dump(preds, open(OUT, "w"))
print("done", len(preds))
