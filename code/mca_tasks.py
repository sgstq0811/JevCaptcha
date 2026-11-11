"""MCA-Bench task registry: official test split (last 200 of the official tojson output) and the official
scoring rules copied from MCA-Bench/test/*.py. Parsing is tolerant: an unparsable answer counts as a failure
(the official scripts raise on it)."""
import json, os, re

ROOT = os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/mca_json")

# name -> (tojson work dir, json file, scorer, official resize-to-224 flag)
TASKS = {
    "color":      ("_work_colortojson", "data_click_color.json", "first_region", False),
    "inverted":   ("_work_dutojson", "data_upside_down_click.json", "all_regions", True),
    "hollow":     ("_work_frettojson", "fretwork_data.json", "all_regions", False),
    "geometric":  ("_work_geotojson", "data_click_Geo.json", "all_regions", False),
    "brightness": ("_work_lighttojson", "light_data.json", "all_regions", False),
    "rotated":    ("_work_rotatetojson", "data_rotate_letter.json", "all_regions", False),
    "vowel":      ("_work_voweltojson", "vowel_data.json", "all_regions", False),
    "fullgrid":   ("_work_holegridtosjon", "data_vl_holegrid.json", "label_set_sorted", True),
    "classic":    ("_work_characterstojson", "data_vl_character.json", "exact_insensitive", False),
    "word":       ("_work_wordtojson", "word.json", "exact", False),
}


FORMAT = {
    "first_region": " Answer only with the pixel coordinates of the point to click, in the format (x,y).",
    "all_regions": " Answer only with the pixel coordinates of every point to click, in the format (x,y); (x,y).",
    "label_set_sorted": " Answer only with the indices of the grid cells to click, separated by commas.",
    "exact_insensitive": " Answer only with the characters.",
    "exact": " Answer only with the answer text.",
}


def load_split(name):
    d, f, _, _ = TASKS[name]
    data = json.load(open(os.path.join(ROOT, d, f), encoding="utf-8"))
    return data[:-200], data[-200:]


def parse_item(item):
    u = item["conversations"][0]["value"]
    img = u.split("<tool_response>")[1]
    img = re.split(r"</tool_response>|</tool_call>", img)[0].strip()
    prompt = u.split("<tool_response>")[0].strip()
    return img, prompt, item["conversations"][1]["value"].strip()


PT = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def pts(s):
    return [(int(float(a)), int(float(b))) for a, b in PT.findall(s)]


def inbox(x, y, b):
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def score(name, item, pred):
    kind = TASKS[name][2]
    gold = item["conversations"][1]["value"].strip()
    if kind == "first_region":                       # test_color.py
        p = pts(pred)
        return len(p) == 1 and inbox(*p[0], item["regions"][0])
    if kind == "all_regions":                        # test_fret/geo/light/rotate/vowel/du
        p, r = pts(pred), item["regions"]
        return len(p) == len(r) and all(any(inbox(x, y, b) for b in r) for x, y in p)
    if kind == "label_set_sorted":                   # testholegrid.py
        return sorted(pred.replace(",", " ").split()) == sorted(gold.replace(",", " ").split())
    if kind == "exact_insensitive":                  # test.py (reports both; case-insensitive is the looser)
        return pred.strip().lower() == gold.lower()
    if kind == "exact":                              # test_math.py style
        return pred.strip() == gold
    raise ValueError(kind)
