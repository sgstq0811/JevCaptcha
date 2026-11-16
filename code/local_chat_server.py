"""OpenAI-compatible /v1/chat/completions for a local generative VLM (greedy, thinking off), so the step-wise
API agent (ocw_api_agent.py through delegate.py) runs unchanged on an open model: LOCAL_MODEL, PORT."""
import os, io, json, base64, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

NAME = os.environ.get("LOCAL_MODEL", "Qwen/Qwen3.5-9B")
proc = AutoProcessor.from_pretrained(NAME)
model = AutoModelForImageTextToText.from_pretrained(NAME, torch_dtype=torch.bfloat16, device_map="cuda").eval()
lock = Lock()


@torch.inference_mode()
def chat(messages, max_new=int(os.environ.get("MAX_NEW", 512))):
    ims, msgs = [], []
    for m in messages:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        c = []
        for p in parts:
            if p["type"] == "image_url":
                ims.append(Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))).convert("RGB"))
                c.append({"type": "image"})
            else:
                c.append({"type": "text", "text": p["text"]})
        msgs.append({"role": m["role"], "content": c})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], images=ims or None, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    L = inp["input_ids"].shape[1]
    return proc.tokenizer.decode(out[0, L:], skip_special_tokens=True), L, out.shape[1] - L


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            txt, a, b = chat(req["messages"])
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": txt}}],
                           "usage": {"prompt_tokens": a, "completion_tokens": b}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", 8011))), H).serve_forever()
