"""Collect the per-candidate typed decisions needed by the analysis:
- Rotation_Match: P(direction) for the reference and for each of the 8 rotated states
- Bingo: pairwise P(same) for all tile pairs
Saved raw so that every aggregation rule (and its subsampled variants) is evaluated offline."""
import os, sys, json, itertools, argparse
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(__file__))
from ocw_policy import Policy, DIRS, choice, grid_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/OpenCaptchaWorld/captcha_data"))
ap.add_argument("--model", default="Cloudflare/clef-flash")
ap.add_argument("--out", required=True)
a = ap.parse_args()
D = a.data
pol = Policy(a.model)
C = pol.clef


def load(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


out = {"rotation": [], "bingo": []}
gt = json.load(open(f"{D}/Rotation_Match/ground_truth.json"))
for pid, g in gt.items():
    base = os.path.splitext(g["object_base_image"])[0]
    st = {"puzzle_instruction": g["prompt"]}
    r = C.decide([pol.rec(st, [load(f"{D}/Rotation_Match/{g['reference_image']}")], {"d": choice("In which direction does the hand or arrow point?", DIRS)})])[0]["d"]
    cs = [c["d"] for c in C.decide([pol.rec(st, [load(f"{D}/Rotation_Match/{base}_{x}.png")], {"d": choice("In which direction does the top of the character's head point?", DIRS)}) for x in range(0, 360, 45)])]
    out["rotation"].append({"id": pid, "ref": r, "states": cs, "gold": (g["correct_angle"] % 360) // 45})
gt = json.load(open(f"{D}/Bingo/ground_truth.json"))
for pid, g in gt.items():
    img = load(f"{D}/Bingo/{pid}")
    tiles = [img.crop(b) for b in grid_boxes(*img.size, 3, 3)]
    pairs = list(itertools.combinations(range(9), 2))
    p = C.noul([({"task": "Compare two tiles of a picture grid."}, [tiles[i], tiles[j]]) for i, j in pairs],
               "Do image 1 and image 2 show the same picture (the same object or animal in the same style)?")
    out["bingo"].append({"id": pid, "pairs": pairs, "p": p, "gold": g["answer"]})
json.dump(out, open(a.out, "w"))
print("done", len(out["rotation"]), len(out["bingo"]))
