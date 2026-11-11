"""Run the Clef decision policy (ocw_policy.Policy) on COGNITION tasks and write answers in the artifact's
parsed-answer format; scoring is done separately with the artifact's evaluate_pass1 (cog_tasks.py score)."""
import os, sys, json, argparse
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(__file__))
from ocw_policy import Policy, grid_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/results/cog_tasks.json"))
ap.add_argument("--split", default="full")
ap.add_argument("--model", default="Cloudflare/clef-flash")
ap.add_argument("--out", required=True)
a = ap.parse_args()
DATA = os.environ.get("COG_DATA", os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/cognition/cognition-usenixsec26-artifact/captcha_data"))
GTS = {}


def gt_entry(t, pid):
    d = {"Connect_Icon": "Connect_icon"}.get(t, t)
    if d not in GTS:
        GTS[d] = json.load(open(f"{DATA}/{d}/ground_truth.json"))
    return GTS[d][pid]


def load(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def base_prompt(p):
    for cut in (" Indexing Rules", "\nIndexing Rules", " The 3x3 grid cells", "\nReturn", " Return JSON"):
        if cut in p:
            p = p.split(cut)[0]
    return p.strip()


pol = Policy(a.model)
tasks = json.load(open(a.tasks))[a.split]
preds = json.load(open(a.out)) if os.path.exists(a.out) else {}
for d in tasks:
    if d["key"] in preds:
        continue
    t, ims = d["type"], d["images"]
    obs = {"prompt": base_prompt(d["prompt"])}
    try:
        if t in ("Connect_Icon", "Coordinates", "Dart_Count", "Image_Matching", "Path_Finder", "Object_Match"):
            obs.update(type={"Connect_Icon": "Connect_icon"}.get(t, t), reference=load(ims[0]), candidates=[load(p) for p in ims[1:]])
            ans = {"answer_type": "classify", "index": int(pol(obs))}
        elif t == "Rotation_Match":
            obj = Image.open(ims[1]).convert("RGBA")
            cands = []
            for ang in range(0, 360, 45):              # the widget turns the object clockwise in 45-degree steps
                r = obj.rotate(-ang, expand=True)
                bg = Image.new("RGB", r.size, (255, 255, 255))
                bg.paste(r, mask=r.split()[3])
                cands.append(bg)
            obs.update(type=t, reference=load(ims[0]), candidates=cands)
            ans = {"answer_type": "rotation", "angle": 45.0 * pol(obs)}
        elif t == "Image_Recognition":
            obs.update(type=t, cells=[load(p) for p in ims])
            ans = {"answer_type": "multi_select", "indices": pol(obs)}
        elif t in ("Unusual_Detection", "Bingo"):
            img = load(ims[0])
            r, c = gt_entry(t, d["pid"]).get("grid_size", [3, 3])
            obs.update(type=t, cells=[img.crop(b) for b in grid_boxes(*img.size, r, c)])
            v = pol(obs)
            ans = {"correct": v} if t == "Bingo" else {"answer_type": "multi_select", "indices": v}
        elif t == "Patch_Select":
            r, c = d["gt"]["grid_size"]
            obs.update(type=t, image=load(ims[0]), n=r * c)
            ans = {"answer_type": "multi_select", "indices": pol(obs)}
        elif t == "Select_Animal":
            r, c = gt_entry(t, d["pid"]).get("grid_size", [2, 3])
            obs.update(type=t, image=load(ims[0]), grid=(r, c))
            ans = {"answer_type": "multi_select", "indices": [pol(obs)]}
        elif t in ("Geometry_Click", "Pick_Area", "Place_Dot", "Misleading_Click"):
            img = load(ims[0])
            obs.update(type=t, image=img, display=Image.open(ims[0]).size)
            x, y = pol(obs)
            ans = {"answer_type": "single_point", "point": {"x": x, "y": y}}
        elif t == "Click_Order":
            img = load(ims[1])
            obs.update(type=t, order=load(ims[0]), image=img, display=Image.open(ims[1]).size)
            ans = {"answer_type": "multi_point", "points": [{"x": x, "y": y} for x, y in pol(obs)]}
        elif t == "Dice_Count":
            obs.update(type=t, image=load(ims[0]))
            ans = {"answer_type": "number", "value": int(pol(obs))}
    except Exception as e:
        ans = None
        print("ERR", d["key"], type(e).__name__, e, flush=True)
    preds[d["key"]] = ans
    json.dump(preds, open(a.out, "w"))
print("done", len(preds))
