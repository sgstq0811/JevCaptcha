"""Typed-decision policy for the Halligan benchmark (26 CAPTCHA types). Input: one official challenge record;
output: the `state` the official frontend would POST to /submit. Every perception is a Clef decision; the
aggregation rules are the ones used for Open CaptchaWorld (verify + argmax, typed matching, pairwise relations
with exhaustive swap search, render-and-verify for continuous answers)."""
import os, re, string, itertools
import numpy as np
from PIL import Image, ImageDraw
from ocw_policy import Policy, DIRS, NUM, choice, argmax, match, grid_boxes
from hal_data import img


def icon_img(b64):
    """transparent icon composited on white, as it appears on the page"""
    a = rgba(b64)
    o = Image.new("RGB", a.size, (255, 255, 255))
    o.paste(a, (0, 0), a)
    return o


def rgba(b64):
    import io, base64
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


def lines(n, diag):
    ls = [[(r, c) for c in range(n)] for r in range(n)] + [[(r, c) for r in range(n)] for c in range(n)]
    if diag:
        ls += [[(i, i) for i in range(n)], [(i, n - 1 - i) for i in range(n)]]
    return ls


class HalPolicy(Policy):
    trace = []

    def noul(self, st, items, instr):
        p = super().noul(st, items, instr)
        self.trace.append([round(x, 4) for x in p])
        return p

    def __call__(self, t, c):
        self.trace = []
        st = {"captcha": t, "instruction": re.sub(r"\*", "", c.get("instruction") or "")}
        name = t.split("/")[-1]
        f = getattr(self, "do_" + name, None) or getattr(self, "do_" + t.split("/")[1], None)
        return f(c, st)

    # ------------------------------------------------------------ arkose multichoice: 6 tiles in a 3x2 sprite
    TILE_Q = {
        "galaxies": "Is the galaxy in this picture a spiral galaxy?",
        "rotated": "Is this picture the correct way up (upright, not rotated)?",
        "square_icon": "Does this picture show two identical objects?",
        "dice_pair": "Do the two dice in this picture show the same icon on their top faces?",
        "card": "Do the two cards in this picture match each other?",
    }

    def tiles6(self, c):
        im = img(c["instruction_image"][0])
        return [im.crop(b) for b in grid_boxes(*im.size, 2, 3)]

    def do_multichoice(self, c, st):
        t = c["subtype"]
        ts = self.tiles6(c)
        if t in self.TILE_Q:
            return argmax(self.noul(st, [[x] for x in ts], self.TILE_Q[t]))
        if t == "hand_number":
            n = re.search(r"(\d+)", c["instruction"]).group(1)
            return argmax(self.noul(st, [[x] for x in ts], f"Count the fingers held up on all hands in this picture. Do they add up to exactly {n}?"))
        if t == "counting":
            q = {"num": choice("What number is written in this picture?", NUM(21)),
                 "cnt": choice("How many animals are in this picture?", NUM(21))}
            ds = self.clef.decide([self.rec(st, [x], q) for x in ts])
            return argmax([sum(d["num"][v] * d["cnt"][v] for v in d["num"]) for d in ds])

    # ------------------------------------------------------------ arkose paged: reference + candidate pages
    def paged_match(self, c, st, qr, qc, opts, ref_in_cand=False):
        ref = img(c["instruction_image"][0])
        cands = [img(b) for b in c["images"]]
        r = self.clef.decide([self.rec(st, [ref], {"a": choice(qr, opts)})])[0]["a"]
        cs = self.clef.decide([self.rec(st, [x, ref] if ref_in_cand else [x], {"a": choice(qc, opts)}) for x in cands])
        return argmax(match(r, [d["a"] for d in cs]))

    def paged_verify(self, c, st, qr, qv, ref_in_cand=False, top=2):
        """read the target number from the reference, then verify 'equals n' on every page for the most likely n;
        page score = sum_n P_ref(n) P(page equals n)"""
        ref = img(c["instruction_image"][0])
        cands = [img(b) for b in c["images"]]
        r = self.clef.decide([self.rec(st, [ref], {"a": choice(qr, NUM(51))})])[0]["a"]
        ns = sorted(r, key=r.get, reverse=True)[:top]
        sc = [0.0] * len(cands)
        for n in ns:
            p = self.noul(st, [[x, ref] if ref_in_cand else [x] for x in cands], qv.format(n=n))
            sc = [s + r[n] * v for s, v in zip(sc, p)]
        return argmax(sc)

    def do_dice_match(self, c, st):
        return self.paged_match(c, st, "What number is shown in this image?",
                                "What is the sum of the values on the top faces of all the dice in this image?", NUM(51))

    def do_rockstack(self, c, st):
        return self.paged_match(c, st, "What number is shown in this image?",
                                "How many rocks are stacked together in the group of rocks in this image?", NUM(11))

    def do_numbermatch(self, c, st):
        return self.paged_verify(c, st, "What number is shown in this image (before the x)?",
                                 "Image 2 shows a number and an icon. Does image 1 contain exactly {n} copies of that icon (any size or rotation)?", True)

    def do_orbit_match_game(self, c, st):
        return self.paged_match(c, st, "What number is shown in this image?",
                                "Image 2 shows an icon. In image 1 the orbits are labelled with numbers on the left. "
                                "What is the number of the orbit on which that icon sits?", NUM(21), True)

    def do_3d_rollball_objects(self, c, st):
        return self.paged_match(c, st, "In which direction does the hand point?",
                                "In which direction does the front of the object face?", DIRS)

    # ------------------------------------------------------------ grid selection
    def do_recaptchav2(self, c, st):
        tgt = c["instruction"]
        if c["subtype"] == "binary":
            p = self.noul(st, [[img(b)] for b in c["images"]], f"Does this picture contain {tgt}?")
            return [x > 0.5 for x in p]
        im = img(c["images"][0])
        p = self.noul(st, [[im, im.crop(b)] for b in grid_boxes(*im.size, 4, 4)],
                      f"Image 2 is one square of image 1 (cut into a 4x4 grid). Does image 2 contain any part of {tgt}?")
        return [False] * 4 + [x > 0.5 for x in p]

    def do_hcaptcha(self, c, st):
        if c["subtype"] in ("hcaptcha_type_2", "hcaptcha_type_3"):
            ims = [img(b) for b in c["images"]]
            if c.get("instruction_image"):
                ex = img(c["instruction_image"][0])
                p = self.noul(st, [[x, ex] for x in ims], "Image 2 is the example pattern. Is the entity in image 1 similar to the example pattern?")
            else:
                p = self.noul(st, [[x] for x in ims], f"The instruction is: '{c['instruction']}'. Should this picture be selected?")
            return self.gap_select(p)
        im = img(c["images"][0])
        x, y = self.mark(im, st, f"The instruction is: '{c['instruction']}'. Is the red dot on the correct target?")
        return [x / im.size[0], y / im.size[1]]

    @staticmethod
    def gap_select(p):
        """tiles above the largest gap of the sorted log-odds: the positives form the upper mode of the puzzle's
        probabilities even when their absolute level is below 0.5"""
        lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        o = np.argsort(-lo)
        k = 1 + int(np.argmax(lo[o][:-1] - lo[o][1:]))
        s = [False] * len(p)
        for i in o[:k]:
            s[int(i)] = True
        return s

    # ------------------------------------------------------------ clicks: render a dot, verify, refine
    def mark(self, im, st, q):
        def render(x, y):
            o = im.copy()
            r = max(5, im.size[0] // 60)
            d = ImageDraw.Draw(o)
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(255, 255, 255))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
            return o
        return self.render_verify(im, st, q, render)

    def do_amazon(self, c, st):
        im = img(c["instruction_image"][0])
        x, y = self.mark(im, st, "A red dot with a white ring has been placed on the picture. Is the red dot exactly at the end of the car's path (the far end of the orange route, away from the car)?")
        return [x / im.size[0], y / im.size[1]]

    def do_tencent(self, c, st):
        im = img(c["instruction_image"][0])
        x, y = self.mark(im, st, f"A red dot with a white ring has been placed on the picture. The instruction is: '{c['instruction']}'. Is the red dot on the correct object?")
        return [x / im.size[0], y / im.size[1]]

    def do_icon(self, c, st):
        """(1) Noul 'one whole icon here' on a lattice of windows, the top windows after non-maximum suppression
        are the icon locations; (2) pairwise Noul 'same symbol' between locations and reference icons;
        (3) the assignment (permutation) with the highest product of pair probabilities"""
        im = img(c["images"][0])
        icons = [icon_img(b) for b in c["instruction_images"]]
        W, H = im.size
        w, h = 0.18 * W, 0.26 * H
        cents = [(x, y) for y in np.arange(h / 2, H - h / 2 + 1, 0.04 * H) for x in np.arange(w / 2, W - w / 2 + 1, 0.03 * W)]
        crop = lambda x, y: im.crop((round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2)))
        p = self.noul(st, [[crop(x, y)] for x, y in cents], "Is there one complete icon (a flat coloured symbol) in the centre of this picture?")
        locs = []
        for k in np.argsort(p)[::-1]:
            x, y = cents[k]
            if all(abs(x - u) > 0.6 * w or abs(y - v) > 0.6 * h for u, v in locs):
                locs.append((x, y))
            if len(locs) == len(icons) + 1:
                break
        pairs = [(i, j) for i in range(len(icons)) for j in range(len(locs))]
        s = self.noul(st, [[icons[i], crop(*locs[j])] for i, j in pairs],
                      "Image 1 is a reference icon. Does image 2 contain the same symbol (colour, size and rotation may differ)?")
        S = {pr: v for pr, v in zip(pairs, s)}
        perm = max(itertools.permutations(range(len(locs)), len(icons)),
                   key=lambda pm: sum(np.log(S[i, j] + 1e-6) for i, j in enumerate(pm)))
        return [[locs[j][0] / W, locs[j][1] / H] for j in perm]

    # ------------------------------------------------------------ continuous drags
    def do_slide(self, c, st):
        bg = img(c["instruction_image"][0])
        pc = rgba(c["images"][0])
        W = bg.size[0]
        y = c["height"]
        xs = [W * k / 100 for k in range(0, 100 - round(100 * pc.size[0] / W) + 1)]

        def render(x):
            o = bg.copy()
            o.paste(pc, (round(x), y), pc)
            return o
        p = self.clef.noul([(st, [render(x)]) for x in xs],
                           "A puzzle piece has been placed on the picture. Does the piece sit exactly in its gap, so the picture looks complete and seamless?")
        return xs[argmax(p)] / W

    def do_baidu(self, c, st):
        im = img(c["instruction_image"][0])
        angs = list(range(0, 360, 10))
        p = self.clef.noul([(st, [im.rotate(-a, resample=Image.BICUBIC, fillcolor=(255, 255, 255))]) for a in angs],
                           "Is this photo upright, as it would naturally be seen (sky and ceilings at the top, ground at the bottom)?")
        return float(angs[argmax(p)])

    def do_kaleidoscope(self, c, st):
        im = img(c["instruction_image"][0])
        tiles = [im.crop(b) for b in grid_boxes(*im.size, 4, 4)]
        tw, th = tiles[0].size
        state = list(range(16))
        renders = []
        for k in range(len(c["swaps"]) + 1):
            if k:
                a, b = c["swaps"][k - 1]
                state[a], state[b] = state[b], state[a]
            o = Image.new("RGB", (tw * 4, th * 4))
            for pos, t in enumerate(state):
                o.paste(tiles[t], ((pos % 4) * tw, (pos // 4) * th))
            renders.append(o)
        p = self.clef.noul([(st, [r]) for r in renders],
                           "This picture is made of 16 tiles. Are all tiles in their correct places, so the picture looks like one complete, continuous photo?")
        return argmax(p)

    def do_lemin(self, c, st):
        im = img(c["instruction_image"][0])
        bg = im.crop((0, 0, 400, 400))
        sc = 1180 / 1156.4
        w = 117.6 * sc
        pieces = []
        for k in range(c["n_pieces"]):
            x0, y0 = (396.9 + k * 127.4) * sc, 4.9 * sc
            pcs = im.crop((round(x0), round(y0), round(x0 + w), round(y0 + w)))
            mask = Image.fromarray(((np.asarray(pcs).sum(2) > 30) * 255).astype("uint8"))
            pieces.append((pcs, mask))
        # which piece fits the hole
        pk = self.noul(st, [[bg, p] for p, _ in pieces], "Image 1 has a hole in it. Is image 2 the puzzle piece whose shape fits that hole?")
        k = argmax(pk)
        pcs, mask = pieces[k]
        g = 9.8 * sc

        def render(x, y):
            o = bg.copy()
            o.paste(pcs, (round(x - w / 2), round(y - w / 2)), mask)
            return o
        x, y = self.render_verify(bg, st, "A puzzle piece has been placed on the picture. Does the piece sit exactly in its hole, so the picture looks complete and seamless?", render)
        return [[k, round(x / g), round(y / g)]]

    # ------------------------------------------------------------ swap puzzles: pairwise equality + exhaustive search
    def swap_search(self, c, st, n, diag, nonempty):
        grid = c["grid"]
        cells = [(r, q) for r in range(n) for q in range(n)]
        ims = {rc: img(c["images"][grid[rc[0]][rc[1]]]) if c["images"][grid[rc[0]][rc[1]]] else None for rc in cells}
        live = [rc for rc in cells if ims[rc] is not None]
        pairs = list(itertools.combinations(live, 2))
        p = self.noul({"task": "Compare two icons of a match puzzle."}, [[ims[a], ims[b]] for a, b in pairs],
                      "Do image 1 and image 2 show the same icon?")
        same = {}
        for (a, b), v in zip(pairs, p):
            same[a, b] = same[b, a] = v
        for a in cells:
            same[a, a] = 1.0
        L = lines(n, diag)
        best, arg = -1, None
        for a, b in itertools.combinations(cells, 2):
            pos = {rc: rc for rc in cells}
            pos[a], pos[b] = b, a
            for ln in L:
                src = [pos[x] for x in ln]
                if nonempty and any(ims[s] is None for s in src):
                    continue
                s = 1.0
                for u, v in itertools.combinations(src, 2):
                    s *= same.get((u, v), 0.0)
                if s > best:
                    best, arg = s, (a, b)
        (r1, c1), (r2, c2) = arg
        g = [row[:] for row in grid]
        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
        return g

    def do_iconcrush(self, c, st):
        return self.swap_search(c, st, 3, False, False)

    def do_gobang(self, c, st):
        return self.swap_search(c, st, 5, True, True)

    # ------------------------------------------------------------ text: length, then one choice per character
    def read_text(self, c, st, chars, maxlen, what="text", lexicon=None, extra=()):
        """length hypotheses from a choice question; for each, one choice per equal-width slice of the text span.
        The candidate readings (and their copies with repeated neighbours merged) are verified on the whole image."""
        im = img(c["instruction_image"][0])
        if im.size[0] < 600:
            im = im.resize((600, round(600 * im.size[1] / im.size[0])), Image.LANCZOS)
        L = self.clef.decide([self.rec(st, [im], {"n": choice(f"How many characters long is the {what} in this image?", NUM(maxlen + 1))})])[0]["n"]
        W, H = im.size
        x0, x1 = self.text_span(im)
        cands = []
        for n in sorted((int(k) for k in L if int(k) > 0), key=lambda k: -L[str(k)])[:3]:
            recs = []
            for i in range(n):
                a, b = x0 + (x1 - x0) * i / n, x0 + (x1 - x0) * (i + 1) / n
                box = im.copy()
                ImageDraw.Draw(box).rectangle([a, 0, b, H - 1], outline=(255, 0, 0), width=2)
                crop = im.crop((max(0, round(a - 0.15 * (b - a))), 0, min(W, round(b + 0.15 * (b - a))), H))
                recs.append(self.rec(st, [box, crop], {"c": choice(f"Image 2 is the part of the {what} inside the red box of image 1. Which character is inside the red box?", chars)}))
            ds = [d["c"] for d in self.clef.decide(recs)]
            txt = "".join(max(d, key=d.get) for d in ds)
            cands.append(txt)
            cands.append("".join(ch for k, ch in enumerate(txt) if k == 0 or ch != txt[k - 1]))
            if lexicon:
                cands += self.lexicon_decode(ds, lexicon)
        cands += list(extra)                            # readings proposed by a tool (generic skill library)
        cands = list(dict.fromkeys(cands))
        # verification of every candidate reading on the whole image
        v = self.clef.decide(
            [self.rec(st, [im], {"v": {"type": "noul", "instructions": f"Does the {what} in this image read exactly '{s}' (ignoring upper/lower case)?"}}) for s in cands])
        pv = [d["v"]["true"] for d in v]
        self.trace.append([cands, [round(x, 4) for x in pv]])
        return cands[argmax(pv)]

    @staticmethod
    def lexicon_decode(ds, lexicon, top=4):
        """MAP two-word readings: the per-slice character distributions scored against every split into two
        dictionary words of the right lengths"""
        n = len(ds)
        lp = [{ch: np.log(v + 1e-6) for ch, v in d.items()} for d in ds]
        out = []
        for k in range(2, n - 1):
            w1 = sorted(lexicon.get(k, []), key=lambda w: -sum(lp[i].get(ch, -14) for i, ch in enumerate(w)))[:top]
            w2 = sorted(lexicon.get(n - k, []), key=lambda w: -sum(lp[k + i].get(ch, -14) for i, ch in enumerate(w)))[:top]
            for a in w1:
                for b in w2:
                    out.append((sum(lp[i].get(ch, -14) for i, ch in enumerate(a + b)), a + " " + b))
        return [s for _, s in sorted(out, reverse=True)[:2 * top]]

    def text_span(self, im):
        """horizontal extent of the text: columns that differ from the background colour"""
        a = np.asarray(im.convert("L"), dtype=np.float32)
        bgv = np.median(np.concatenate([a[:, :3].ravel(), a[:, -3:].ravel()]))
        col = (np.abs(a - bgv) > 60).mean(0)
        idx = np.where(col > 0.03)[0]
        return (int(idx[0]), int(idx[-1]) + 1) if len(idx) > 5 else (0, im.size[0])

    def do_botdetect(self, c, st):
        return self.read_text(c, st, list(string.ascii_uppercase + string.digits), 8)

    def do_mtcaptcha(self, c, st):
        return self.read_text(c, st, list(string.ascii_lowercase + string.digits), 8)

    def do_text(self, c, st):
        if not hasattr(self, "lex"):
            self.lex = {}
            for w in open(os.path.expanduser(os.environ.get("JEVCAPTCHA_ROOT", "~/captcha_jev") + "/data/lexicon/american-english")):
                w = w.strip()
                if w.isalpha() and w.islower() and w.isascii():
                    self.lex.setdefault(len(w), []).append(w)
        im = img(c["instruction_image"][0])
        x0, x1 = self.text_span(im)
        cut = self.word_gap(im, x0, x1)
        return "".join(self.read_word(im.crop((a, 0, b, im.size[1])), st) for a, b in ((x0, cut), (cut, x1)))

    def word_gap(self, im, x0, x1):
        """the widest run of near-empty columns in the middle part of the text span"""
        a = np.asarray(im.convert("L"), dtype=np.float32)
        bgv = np.median(np.concatenate([a[:, :3].ravel(), a[:, -3:].ravel()]))
        ink = (np.abs(a - bgv) > 60).mean(0)
        lo, hi = x0 + (x1 - x0) // 5, x1 - (x1 - x0) // 5
        best, run, start, arg = 0, 0, lo, (x0 + x1) // 2
        for x in range(lo, hi):
            if ink[x] < 0.02:
                run += 1
                if run == 1:
                    start = x
                if run > best:
                    best, arg = run, start + run // 2
            else:
                run = 0
        return arg

    def read_word(self, im, st):
        """one dictionary word: equal-slice readings for the top length hypotheses, lexicon decoding of each,
        Noul verification of all candidates on the word image"""
        if im.size[0] < 400:
            im = im.resize((400, round(400 * im.size[1] / im.size[0])), Image.LANCZOS)
        W, H = im.size
        chars = list(string.ascii_lowercase)
        L = self.clef.decide([self.rec(st, [im], {"n": choice("How many letters long is the word in this image?", NUM(17))})])[0]["n"]
        cands = []
        for n in sorted((int(k) for k in L if int(k) > 1), key=lambda k: -L[str(k)])[:4]:
            recs = []
            for i in range(n):
                a, b = W * i / n, W * (i + 1) / n
                box = im.copy()
                ImageDraw.Draw(box).rectangle([a, 0, b, H - 1], outline=(255, 0, 0), width=2)
                crop = im.crop((max(0, round(a - 0.15 * (b - a))), 0, min(W, round(b + 0.15 * (b - a))), H))
                recs.append(self.rec(st, [box, crop], {"c": choice("Image 2 is the part of the word inside the red box of image 1. Which letter is inside the red box?", chars)}))
            ds = [d["c"] for d in self.clef.decide(recs)]
            lp = [{ch: np.log(v + 1e-6) for ch, v in d.items()} for d in ds]
            ws = sorted(self.lex.get(n, []), key=lambda w: -sum(lp[i][ch] for i, ch in enumerate(w)))[:6]
            cands += ws or ["".join(max(d, key=d.get) for d in ds)]
        cands = list(dict.fromkeys(cands))
        v = self.clef.decide([self.rec(st, [im], {"v": {"type": "noul", "instructions": f"Does the word in this image read exactly '{s}'?"}}) for s in cands])
        pv = [d["v"]["true"] for d in v]
        self.trace.append([cands, [round(x, 4) for x in pv]])
        return cands[argmax(pv)]
