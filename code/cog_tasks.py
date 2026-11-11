"""COGNITION official task sampling and scoring (runs in the artifact's own venv).
  dump : build the main-benchmark tasks with the artifact's build_tasks (seed 1234, max 20 per type, the type order
         of results/exp2 rows) and also the full set (no cap); write them as JSON
  score: score a predictions file ({key: parsed_answer_dict}) with the artifact's evaluate_pass1"""
import sys, os, json, random, collections
ROOT = os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/cognition/cognition-usenixsec26-artifact")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from cognition import run_eval as R

TYPES = ["Bingo", "Click_Order", "Connect_Icon", "Coordinates", "Dart_Count", "Dice_Count", "Geometry_Click", "Image_Matching",
         "Image_Recognition", "Patch_Select", "Path_Finder", "Place_Dot", "Rotation_Match", "Select_Animal", "Unusual_Detection",
         "Misleading_Click", "Object_Match", "Pick_Area"]


def ser(tasks):
    return [{"key": f"{t.type}/{t.puzzle_id}", "type": t.type, "pid": t.puzzle_id, "prompt": t.prompt,
             "images": [os.path.abspath(p) for p in t.images], "gt": t.gt} for t in tasks]


if sys.argv[1] == "dump":
    random.seed(1234)
    main = R.build_tasks("./captcha_data", TYPES, 20)
    random.seed(1234)
    full = R.build_tasks("./captcha_data", TYPES, None)
    out = sys.argv[2]
    json.dump({"main": ser(main), "full": ser(full)}, open(out, "w"), indent=1, default=list)
    print("main", len(main), dict(collections.Counter(t.type for t in main)))
    print("full", len(full))
else:
    tasks = json.load(open(sys.argv[2]))
    preds = json.load(open(sys.argv[3]))
    for split in ("main", "full"):
        by = collections.defaultdict(lambda: [0, 0])
        for d in tasks[split]:
            t = R.TaskItem(d["type"], d["pid"], d["prompt"], d["images"], d["gt"])
            ok = bool(R.evaluate_pass1(t, preds.get(d["key"])))
            by[d["type"]][0] += ok
            by[d["type"]][1] += 1
        n = sum(v[1] for v in by.values())
        k = sum(v[0] for v in by.values())
        print(f"[{split}] pass@1 {k}/{n} = {100 * k / n:.1f}")
        print("  " + "  ".join(f"{t}:{v[0]}/{v[1]}" for t, v in by.items()))
