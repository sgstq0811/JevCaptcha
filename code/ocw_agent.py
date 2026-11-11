"""Browser agent for Open CaptchaWorld: drives the official web app (Flask + frontend JS) in headless Chromium.
The agent reads only what the page shows (DOM images, prompt text), enumerates candidate actions through the
page's own controls, asks a decision policy, executes the action with real mouse/keyboard events, and records the
official /api/check_answer verdict. Every puzzle of every type is attempted once (Pass@1)."""
import io, os, sys, json, time, argparse
from PIL import Image, ImageOps

URL = os.environ.get("OCW_URL", "http://127.0.0.1:7860")
GT_ROOT = os.environ.get("OCW_GT", os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/OpenCaptchaWorld/captcha_data"))
TYPES = ['Dice_Count', 'Geometry_Click', 'Rotation_Match', 'Slide_Puzzle', 'Unusual_Detection', 'Image_Recognition',
         'Bingo', 'Image_Matching', 'Patch_Select', 'Dart_Count', 'Object_Match', 'Select_Animal', 'Coordinates',
         'Path_Finder', 'Place_Dot', 'Connect_icon', 'Click_Order', 'Hold_Button', 'Misleading_Click', 'Pick_Area']
# cycle widgets: (right-arrow selector, submit selector, candidate image selector)
CYCLE = {
    "Rotation_Match": (".rotate-right", ".submit-rotation", "#object-image"),
    "Image_Matching": (".navigate-right", ".submit-image-matching", "#option-image"),
    "Coordinates": (".navigate-right", ".submit-image-matching", "#option-image"),
    "Path_Finder": (".navigate-right", ".submit-image-matching", "#option-image"),
    "Object_Match": (".object-match-arrow.right-arrow", ".object-match-submit", "img.object-match-option-img"),
    "Dart_Count": (".dart-count-arrow.right-arrow", ".dart-count-submit", "img.dart-count-option-img"),
    "Connect_icon": (".arrow-btn.right-arrow", ".submit-connect", "#connect-option-image"),
}
REF = {"Rotation_Match": "#reference-image", "Image_Matching": "#reference-image", "Coordinates": "#reference-image",
       "Path_Finder": "#reference-image", "Object_Match": "img.object-match-reference-img",
       "Dart_Count": "img.dart-count-reference-img", "Connect_icon": "#connect-reference-image"}


class Page:
    def __init__(self, pg):
        self.pg = pg

    def image(self, src):
        if src.startswith("data:"):
            import base64
            data = base64.b64decode(src.split(",", 1)[1])
        else:
            data = self.pg.request.get(URL + src if src.startswith("/") else src).body()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")

    def src(self, sel):
        return self.pg.get_attribute(sel, "src")

    def rect(self, sel, i=0):
        return self.pg.locator(sel).nth(i).bounding_box()


def solve(page, t, policy, log):
    pg = page.pg
    prompt = pg.inner_text("#puzzle-prompt")
    obs = {"type": t, "prompt": prompt, "puzzle_id": log["puzzle_id"]}
    if t in CYCLE:
        right, submit, cand = CYCLE[t]
        obs["reference"] = page.image(page.src(REF[t]))
        srcs = []
        while True:                                   # enumerate every state reachable with the arrows
            s = page.src(cand)
            if s in srcs:
                break
            srcs.append(s)
            pg.click(right)
            pg.wait_for_timeout(150)
        obs["candidates"] = [page.image(s) for s in srcs]
        k = policy(obs)
        while page.src(cand) != srcs[k]:
            pg.click(right)
            pg.wait_for_timeout(100)
        log["action"] = k
        pg.click(submit)
    elif t in ("Image_Recognition", "Unusual_Detection", "Bingo"):
        cells = pg.locator(".grid-cell")
        n = cells.count()
        obs["cells"] = [page.image(cells.nth(i).locator("img").get_attribute("src")) for i in range(n)]
        sel = policy(obs)
        for i in sel:
            cells.nth(i).click()
            pg.wait_for_timeout(100)
        log["action"] = sel
        pg.click({"Image_Recognition": ".submit-image-recognition", "Unusual_Detection": ".submit-unusual", "Bingo": ".submit-bingo"}[t])
    elif t == "Patch_Select":
        obs["image"] = page.image(pg.get_attribute(".patch-select-image-container img", "src"))
        cells = pg.locator(".patch-select-cell")
        obs["n"] = cells.count()
        sel = policy(obs)
        for i in sel:
            cells.nth(i).click()
            pg.wait_for_timeout(80)
        log["action"] = sel
        pg.click("#submit-answer")
    elif t == "Select_Animal":
        img = pg.locator(".puzzle-image-container img").first
        obs["image"] = page.image(img.get_attribute("src"))
        cells = pg.locator(".puzzle-image-container img + div > div")
        obs["n"] = cells.count()
        g = pg.evaluate("() => { const d = document.querySelector('.puzzle-image-container img + div'); return [d.style.gridTemplateRows, d.style.gridTemplateColumns]; }")
        obs["grid"] = tuple(int(x.split("(")[1].split(",")[0]) for x in g)
        k = policy(obs)
        cells.nth(k).click()
        log["action"] = k
        pg.click("#submit-answer")
    elif t in ("Geometry_Click", "Pick_Area", "Misleading_Click", "Place_Dot"):
        sel = "#puzzle-image" if t != "Place_Dot" else ".puzzle-image-container img"
        img = pg.locator(sel).first
        obs["image"] = page.image(img.get_attribute("src"))
        box = img.bounding_box()
        obs["display"] = (box["width"], box["height"])
        x, y = policy(obs)                            # displayed CSS pixels inside the image
        img.click(position={"x": x, "y": y})
        log["action"] = [x, y]
        pg.wait_for_timeout(200)
        if pg.locator("#submit-answer").is_visible():
            pg.click("#submit-answer")
    elif t == "Click_Order":
        obs["order"] = page.image(page.src("#click-order-reference-image"))
        main = pg.locator("#click-order-main-image")
        obs["image"] = page.image(main.get_attribute("src"))
        box = main.bounding_box()
        obs["display"] = (box["width"], box["height"])
        pts = policy(obs)
        for x, y in pts:
            main.click(position={"x": x, "y": y})
            pg.wait_for_timeout(150)
        log["action"] = pts
        pg.click(".click-order-submit-btn")
    elif t == "Slide_Puzzle":
        bg = pg.locator(".background-container img").first
        comp = pg.locator(".slider-component")
        obs["image"] = page.image(bg.get_attribute("src"))
        obs["component"] = page.image(comp.locator("img").get_attribute("src"))
        bb, cb = bg.bounding_box(), comp.locator("img").bounding_box()
        obs["display"] = (bb["width"], bb["height"])
        obs["component_display"] = (cb["width"], cb["height"])
        tx, ty = policy(obs)                          # target centre of the component, container CSS pixels
        js = """() => { const s = document.querySelector('.slider-component'); const i = s.querySelector('img').getBoundingClientRect();
                       return [(parseInt(s.style.left) || 0) + i.width / 2, (parseInt(s.style.top) || 0) + i.height / 2]; }"""
        for _ in range(3):                            # drag, read where the page put the piece, correct
            cx, cy = pg.evaluate(js)
            if abs(cx - tx) < 1 and abs(cy - ty) < 1:
                break
            cb = comp.locator("img").bounding_box()
            sx, sy = cb["x"] + cb["width"] / 2, cb["y"] + cb["height"] / 2
            pg.mouse.move(sx, sy)
            pg.mouse.down()
            for i in range(1, 26):
                pg.mouse.move(sx + (tx - cx) * i / 25, sy + (ty - cy) * i / 25)
                pg.wait_for_timeout(15)
            pg.mouse.up()
            pg.wait_for_timeout(100)
        log["action"] = [tx, ty]
        pg.click(".submit-slider")
    elif t == "Hold_Button":
        btn = pg.locator(".hold-button")
        b = btn.bounding_box()
        pg.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        pg.mouse.down()
        t0 = time.time()
        while time.time() - t0 < 8:                   # hold until the page's progress bar is full
            w = pg.evaluate("() => { const p = document.querySelector('.hold-progress'); return p ? p.style.width : ''; }")
            if w.startswith("100"):
                break
            pg.wait_for_timeout(100)
        pg.wait_for_timeout(200)
        pg.mouse.up()
        log["action"] = round(time.time() - t0, 2)
        pg.wait_for_timeout(300)
        if pg.locator("#submit-answer").is_visible():
            pg.click("#submit-answer")
    elif t == "Dice_Count":
        obs["image"] = page.image(page.src("#puzzle-image"))
        v = policy(obs)
        pg.fill("#user-answer", str(v))
        log["action"] = v
        pg.click("#submit-answer")


def run(policy, types, out, limit=0):
    results = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        for t in types:
            n = json.loads(open(f"{GT_ROOT}/{t}/ground_truth.json").read())
            n = len(n) if not limit else min(limit, len(n))
            ctx = b.new_context(viewport={"width": 1280, "height": 1800})
            pg = ctx.new_page()

            def handler(route, t=t, pg=pg):
                route.fulfill(response=pg.request.get(f"{URL}/api/get_puzzle?debug_type={t}"))
            pg.route("**/api/get_puzzle**", lambda route: handler(route))
            ok = 0
            for i in range(n):
                with pg.expect_response(lambda r: "/api/get_puzzle" in r.url) as rp:
                    pg.goto(URL)
                pid = rp.value.json()["puzzle_id"]
                pg.wait_for_timeout(1500)
                log = {"type": t, "puzzle_id": pid}
                try:
                    with pg.expect_response(lambda r: "/api/check_answer" in r.url, timeout=600000) as ra:
                        solve(Page(pg), t, policy, log)
                    v = ra.value.json()
                    log["correct"] = bool(v.get("correct"))
                    log["server"] = v
                except Exception as e:
                    log["correct"] = False
                    log["error"] = f"{type(e).__name__}: {e}"[:300]
                ok += log["correct"]
                results.append(log)
                with open(out, "a") as f:
                    f.write(json.dumps(log) + "\n")
            print(f"{t}: {ok}/{n}", flush=True)
            ctx.close()
        b.close()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="dummy")
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.policy == "oracle":                          # pipeline check only: answers from ground truth
        GT = {t: json.load(open(f"{GT_ROOT}/{t}/ground_truth.json")) for t in TYPES}

        def policy(obs):
            t, g = obs["type"], GT[obs["type"]][obs["puzzle_id"]]
            if t in CYCLE:
                return None
            if t == "Slide_Puzzle":
                return tuple(g["target_position"])
            if t == "Place_Dot":
                return tuple(g["target_position"])
            if t in ("Geometry_Click", "Pick_Area"):
                (x0, y0), (x1, y1) = g["answer"]["area"]
                return ((x0 + x1) / 2, (y0 + y1) / 2)
            if t == "Misleading_Click":
                A = g["avoid_area"]
                w, h = obs["display"]
                return (5, 5) if not (A["x"] <= 5 <= A["x"] + A["width"] and A["y"] <= 5 <= A["y"] + A["height"]) else (w - 5, h - 5)
            if t == "Click_Order":
                return [tuple(p) for p in g["answer"]]
            return None
    elif a.policy == "dummy":
        def policy(obs):
            t = obs["type"]
            if t in CYCLE:
                return 0
            if t in ("Image_Recognition", "Unusual_Detection", "Patch_Select"):
                return [0]
            if t == "Bingo":
                return [0, 1]
            if t == "Select_Animal":
                return 0
            if t == "Click_Order":
                w, h = obs["display"]
                return [(w * .2, h * .2), (w * .8, h * .8)]
            if t in ("Geometry_Click", "Pick_Area", "Misleading_Click", "Place_Dot", "Slide_Puzzle"):
                w, h = obs["display"]
                return (w / 2, h / 2)
            if t == "Dice_Count":
                return 10
    else:
        sys.path.insert(0, os.path.dirname(__file__))
        from ocw_policy import make_policy
        policy = make_policy(a.policy)
    run(policy, a.types, a.out, a.limit)
