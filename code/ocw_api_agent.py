"""Frontier-model agent for Open CaptchaWorld: each step the model ($DELEGATE, through delegate.py) sees the puzzle
screenshot, the same screenshot with the interactive elements outlined and numbered (found from the DOM by kind, as in
ocw_generic.py), the element list, the instruction and its action history, and writes the next action as JSON:
click an element, click a pixel, drag an element to a pixel, type text into an element, press and hold an element,
or stop. The episode ends when the page submits an answer or after MAX_STEPS."""
import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(__file__))
import delegate
from ocw_generic import GenericAgent, FIND, dedupe, run, MAX_STEPS

ACTIONS = ('{"action": "click", "element": k} | {"action": "click_point", "x": px, "y": px} | '
           '{"action": "drag", "element": k, "x": px, "y": px} | {"action": "type", "element": k, "text": "..."} | '
           '{"action": "hold", "element": k} | {"action": "stop"}')


class APIAgent(GenericAgent):
    def __init__(self):                                 # no local model
        self.tools = False

    def step(self, pg, prompt, hist):
        im, box = self.shot(pg)
        els = dedupe(pg.evaluate(FIND))
        lst = "\n".join(f"{i}: {e['tag']}" + (f" '{e['text']}'" if e["text"] else "") + (" (text box)" if e["input"] else "")
                        + (" (draggable)" if e["drag"] else "") + f" at x={e['x'] - box['x']:.0f}..{e['x'] - box['x'] + e['w']:.0f}, "
                        f"y={e['y'] - box['y']:.0f}..{e['y'] - box['y'] + e['h']:.0f}" for i, e in enumerate(els, 1))
        text = (f"You are solving a CAPTCHA on a web page. Instruction: {prompt}\n"
                f"Image 1 is a screenshot of the puzzle ({im.size[0]}x{im.size[1]} pixels); image 2 is the same screenshot "
                f"with the interactive elements outlined and numbered in red.\nElements:\n{lst}\n"
                f"Actions so far: {hist or 'none'}. Steps left: {MAX_STEPS - len(hist)}.\n"
                f"Give the next action (pixel coordinates are in image 1), as one JSON object, one of: {ACTIONS}. "
                f"Remember to submit the answer if the page has a submit control. Output only the JSON.")
        out = delegate.ask([im, self.marked(im, box, els)], text)
        try:
            a = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except Exception:
            return None
        kind = a.get("action")
        e = els[int(a["element"]) - 1] if str(a.get("element", "")).isdigit() and 0 < int(a["element"]) <= len(els) else None
        if kind == "stop" or (kind in ("click", "drag", "type", "hold") and e is None):
            return None
        if kind == "click":
            pg.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
        elif kind == "click_point":
            pg.mouse.click(box["x"] + float(a["x"]), box["y"] + float(a["y"]))
        elif kind == "drag":
            cx, cy, tx, ty = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2, box["x"] + float(a["x"]), box["y"] + float(a["y"])
            pg.mouse.move(cx, cy)
            pg.mouse.down()
            for j in range(1, 26):
                pg.mouse.move(cx + (tx - cx) * j / 25, cy + (ty - cy) * j / 25)
                pg.wait_for_timeout(15)
            pg.mouse.up()
        elif kind == "type":
            pg.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2)
            pg.keyboard.press("Control+A")
            pg.keyboard.type(str(a.get("text", "")))
        elif kind == "hold":
            return json.dumps(a) + ": " + self.hold(pg, e)
        else:
            return None
        pg.wait_for_timeout(300)
        return json.dumps(a)


if __name__ == "__main__":
    from ocw_agent import TYPES
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(APIAgent(), a.types, a.out, a.limit)
