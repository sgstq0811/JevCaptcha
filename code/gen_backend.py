"""Generative-VLM backend with the same interface as clef.Clef (decide / noul), used as the controlled
comparison: identical records (state, images, typed questions) are answered by a generative model, and the
probability of each allowed option is its normalised sequence likelihood as the assistant reply
(option tokens followed by the end-of-turn token), computed from a shared prefix cache."""
import os, copy, json, torch
SCORE = os.environ.get("GEN_SCORE", "seq")   # seq: sequence likelihood; first: first-token logits when the options differ there, else per-token mean
from transformers import AutoProcessor, AutoModelForImageTextToText


class GenVLM:
    def __init__(self, model="Qwen/Qwen3.5-9B", device="cuda", max_side=768):
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.tok = self.proc.tokenizer
        self.max_side = max_side
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _fit(self, im):
        w, h = im.size
        s = self.max_side / max(w, h)
        return im.resize((max(28, round(w * s)), max(28, round(h * s)))) if s < 1 else im

    def _options(self, q):
        if q["type"] == "noul":
            return ["true", "false"], ["yes", "no"]
        crit = q["criteria"]
        if isinstance(crit, dict):
            return list(crit), list(crit)
        return [str(i) for i in range(len(crit))], [str(i) for i in range(len(crit))]

    @torch.inference_mode()
    def _one(self, rec, qid, q):
        ids, words = self._options(q)
        state = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"])
        if q["type"] == "noul":
            ask = f"{q.get('instructions', qid)}\nAnswer with exactly one word: yes or no."
        else:
            ask = f"{q.get('instructions', qid)}\nAnswer with exactly one of: {', '.join(words)}."
        content = [{"type": "image"} for _ in rec.get("images", [])] + [{"type": "text", "text": f"Context: {state}\n{ask}"}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ims = [self._fit(i) for i in rec.get("images", [])] or None
        inp = self.proc(text=[text], images=ims, return_tensors="pt").to(self.model.device)
        out = self.model(**inp, use_cache=True)
        first = out.logits[0, -1].float().log_softmax(-1)
        cache = out.past_key_values
        L = inp["input_ids"].shape[1]
        if SCORE == "first":
            heads = [self.tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
            if len(set(heads)) == len(heads):
                scores = [first[t].item() for t in heads]
                m = max(scores)
                p = [pow(2.718281828, s - m) for s in scores]
                return dict(zip(ids, [x / sum(p) for x in p]))
        scores = []
        for w in words:
            toks = self.tok(w, add_special_tokens=False)["input_ids"] + [self.end]
            lp = first[toks[0]].item()
            if len(toks) > 1:
                c = copy.deepcopy(cache)
                x = torch.tensor([toks[:-1]], device=self.model.device)
                pos = torch.arange(L, L + len(toks) - 1, device=self.model.device)
                o = self.model(input_ids=x, past_key_values=c, use_cache=True, cache_position=pos)
                lps = o.logits[0].float().log_softmax(-1)
                lp += sum(lps[i, t].item() for i, t in enumerate(toks[1:]))
            scores.append(lp / len(toks) if SCORE == "first" else lp)
        m = max(scores)
        p = [pow(2.718281828, s - m) for s in scores]
        z = sum(p)
        return dict(zip(ids, [x / z for x in p]))

    def decide(self, records):
        return [{qid: self._one(r, qid, q) for qid, q in r["questions"].items()} for r in records]

    def noul(self, items, instruction):
        recs = [{"state": st, "images": ims, "questions": {"q": {"type": "noul", "instructions": instruction}}} for st, ims in items]
        return [a["q"]["true"] for a in self.decide(recs)]
