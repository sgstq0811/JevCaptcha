"""Run the official MCA-Bench tojson scripts on the released Kaggle data, then apply the official split
(train = data[:-200], test = data[-200:], see train/train.py).

The official scripts walk directories with os.walk / os.listdir / glob, whose order is filesystem-dependent.
We make the order deterministic by sorting every directory listing; everything else is the official code.
"""
import os, sys, json, glob, runpy, builtins
from pathlib import Path

TOJSON = os.path.abspath(sys.argv[1])           # MCA-Bench/tojson
DATA = os.path.abspath(sys.argv[2])             # .../data_set/dataset
OUT = os.path.abspath(sys.argv[3])
os.makedirs(OUT, exist_ok=True)

# task folder -> (official tojson script, name of the dir variable, output file written by the script)
TASKS = {
    "3×3 grid selection": ("gridtojson.py", "BASE_DIR"),
    "full-image grid selection": ("holegridtosjon.py", "BASE_DIR"),
    "arithmetic‐selection": ("clickmathtojson.py", "BASE_DIR"),
    "color discrimination": ("colortojson.py", "BASE_DIR"),
    "inverted‐letter selection": ("dutojson.py", "BASE_DIR"),
    "hollow‐pattern recognition": ("frettojson.py", "BASE_DIR"),
    "geometric‐shape recognition": ("geotojson.py", "BASE_DIR"),
    "brightness discrimination": ("lighttojson.py", "BASE_DIR"),
    "rotated‐letter selection": ("rotatetojson.py", "BASE_DIR"),
    "vowel selection": ("voweltojson.py", "BASE_DIR"),
    "sequential‐letter ordering": ("tapCsequencetojson.py", "BASE_DIR"),
    "classic character CAPTCHA": ("characterstojson.py", "IMAGE_DIR"),
    "arithmetic‐character CAPTCHA": ("mathtojson.py", "BASE_DIR"),
    "distorted‐word CAPTCHA": ("wordtojson.py", "BASE_DIR"),
    "alignment sliders": ("concattojson.py", "IMAGES_DIR"),
    "rotation‐block": ("rotateblocktojson.py", "IMAGES_DIR"),
    "sliding‐block": ("slidertojson.py", "IMAGES_DIR"),
}

_walk, _listdir, _glob, _iterdir = os.walk, os.listdir, glob.glob, Path.iterdir


def s_walk(top, *a, **k):
    for root, dirs, files in _walk(top, *a, **k):
        dirs.sort()
        yield root, dirs, sorted(files)


os.walk = s_walk
os.listdir = lambda p=".": sorted(_listdir(p))
glob.glob = lambda *a, **k: sorted(_glob(*a, **k))
Path.iterdir = lambda self: iter(sorted(_iterdir(self)))

which = sys.argv[4:] or list(TASKS)
for task in which:
    script, var = TASKS[task]
    src = open(os.path.join(TOJSON, script), encoding="utf-8").read()
    d = os.path.join(DATA, task)
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if l.startswith(var) and "=" in l:
            lines[i] = f"{var} = {d!r}"
            break
    else:
        raise SystemExit(f"{script}: {var} not found")
    tmp = os.path.join(OUT, "_run_" + script)
    open(tmp, "w", encoding="utf-8").write("\n".join(lines))
    cwd = os.getcwd()
    work = os.path.join(OUT, "_work_" + script[:-3])
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    try:
        sys.argv = [tmp]
        runpy.run_path(tmp, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print(f"[FAIL] {task}: {type(e).__name__}: {e}")
        os.chdir(cwd)
        continue
    os.chdir(cwd)
    outs = [f for f in _listdir(work) if f.endswith(".json")]
    for f in outs:
        data = json.load(open(os.path.join(work, f), encoding="utf-8"))
        print(f"{task}: {script} -> {f}: {len(data)} samples")
