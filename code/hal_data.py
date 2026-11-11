"""Halligan benchmark (USENIX Sec'25, Zenodo 15709075): 26 CAPTCHA types x 100 challenges.
Loads the official challenge JSON files and decodes their images; answers are checked by the official Flask server."""
import os, io, json, base64
from PIL import Image

ROOT = os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/data/halligan/bench/benchmark/apis")
# route (as in the README / paper Table 4) -> challenge file, submit URL template
TYPES = {
    "tencent": ("tencent/challenges.json", "/tencent/submit"),
    "mtcaptcha": ("mtcaptcha/challenges.json", "/mtcaptcha/submit"),
    "yandex/text": ("yandex/text/challenges.json", "/yandex/text/submit"),
    "botdetect": ("botdetect/challenges.json", "/botdetect/submit"),
    "recaptchav2": ("recaptchav2/challenges.json", "/recaptchav2/submit"),
    "baidu": ("baidu/challenges.json", "/baidu/submit"),
    "hcaptcha": ("hcaptcha/challenges.json", "/hcaptcha/submit"),
    "geetest/slide": ("geetest/slide/challenges.json", "/geetest/slide/submit"),
    "lemin": ("lemin/challenges.json", "/lemin/submit"),
    "amazon": ("amazon/challenges.json", "/amazon/submit"),
    "arkose/multichoice/counting": ("arkose/multichoice/counting.json", "/arkose/multichoice/counting/{id}/submit"),
    "arkose/multichoice/hand_number": ("arkose/multichoice/hand_number.json", "/arkose/multichoice/hand_number/{id}/submit"),
    "arkose/multichoice/galaxies": ("arkose/multichoice/galaxies.json", "/arkose/multichoice/galaxies/{id}/submit"),
    "arkose/multichoice/dice_pair": ("arkose/multichoice/dice_pair.json", "/arkose/multichoice/dice_pair/{id}/submit"),
    "arkose/multichoice/card": ("arkose/multichoice/card.json", "/arkose/multichoice/card/{id}/submit"),
    "arkose/multichoice/square_icon": ("arkose/multichoice/square_icon.json", "/arkose/multichoice/square_icon/{id}/submit"),
    "arkose/multichoice/rotated": ("arkose/multichoice/rotated.json", "/arkose/multichoice/rotated/{id}/submit"),
    "arkose/paged/3d_rollball_objects": ("arkose/paged/3d_rollball_objects.json", "/arkose/paged/3d_rollball_objects/{id}/submit"),
    "arkose/paged/dice_match": ("arkose/paged/dice_match.json", "/arkose/paged/dice_match/{id}/submit"),
    "arkose/paged/orbit_match_game": ("arkose/paged/orbit_match_game.json", "/arkose/paged/orbit_match_game/{id}/submit"),
    "arkose/paged/rockstack": ("arkose/paged/rockstack.json", "/arkose/paged/rockstack/{id}/submit"),
    "arkose/paged/numbermatch": ("arkose/paged/numbermatch.json", "/arkose/paged/numbermatch/{id}/submit"),
    "geetest/icon": ("geetest/icon/challenges.json", "/geetest/icon/submit"),
    "geetest/iconcrush": ("geetest/iconcrush/challenges.json", "/geetest/iconcrush/submit"),
    "geetest/gobang": ("geetest/gobang/challenges.json", "/geetest/gobang/submit"),
    "yandex/kaleidoscope": ("yandex/kaleidoscope/challenges.json", "/yandex/kaleidoscope/submit"),
}


def img(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def load(t):
    d = json.load(open(os.path.join(ROOT, TYPES[t][0])))
    ch = d["challenges"]
    if "images" in d:  # geetest gobang / iconcrush keep a shared image table
        for c in ch:
            c["images"] = {item: d["images"][item] for row in c["grid"] for item in row}
    return ch[:100]
