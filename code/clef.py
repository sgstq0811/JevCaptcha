"""Thin wrapper around Cloudflare Clef / Clef-flash (Jev-compatible typed decision model)."""
import sys, torch
from huggingface_hub import snapshot_download


class Clef:
    def __init__(self, model="Cloudflare/clef-flash", device="cuda", batch=4, max_side=512):
        path = snapshot_download(model)
        sys.path.insert(0, path)
        from joint_schema_model import collate_records, encode_record, load_release_model
        self._collate, self._encode = collate_records, encode_record
        self.model, self.processor = load_release_model(path, device=device)
        self.device, self.batch, self.max_side = torch.device(device), batch, max_side

    def _fit(self, im):
        w, h = im.size
        s = self.max_side / max(w, h)
        return im.resize((max(28, round(w * s)), max(28, round(h * s)))) if s < 1 else im

    @torch.inference_mode()
    def decide(self, records):
        """records: list of {"state", "images", "questions"} -> list of {qid: {option: prob}}."""
        out = []
        for s in range(0, len(records), self.batch):
            recs = [dict(r, images=[self._fit(i) for i in r.get("images", [])]) for r in records[s:s + self.batch]]
            encs = [self._encode(self.processor.tokenizer, r, processor=self.processor) for r in recs]
            b = self._collate(encs, self.processor.tokenizer.pad_token_id, self.device)
            res = self.model(b)
            for i, enc in enumerate(encs):
                ans = {}
                for q, lg in zip(enc.questions, res[i]):
                    p = lg.float().softmax(-1).tolist()
                    ans[q.question_id] = dict(zip(q.option_ids, p))
                out.append(ans)
        return out

    def noul(self, items, instruction):
        """items: list of (state, images); returns P(true) per item."""
        recs = [{"state": st, "images": ims, "questions": {"q": {"type": "noul", "instructions": instruction}}} for st, ims in items]
        return [a["q"]["true"] for a in self.decide(recs)]
