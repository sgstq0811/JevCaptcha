"""Direct-prompting baseline on the MCA-Bench official test split with a local generative VLM: the official instruction
with the answer format appended (mca_tasks.FORMAT) and the image go to the model, which writes its answer; graded with
the official rules (mca_tasks.score). No skills, no enumeration."""
import os, sys, json, argparse, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
sys.path.insert(0, os.path.dirname(__file__))
import mca_tasks as M

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--tasks", default=",".join(M.TASKS))
ap.add_argument("--out", required=True)
a = ap.parse_args()
proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
done = {(r["task"], r["i"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
with open(a.out, "a") as f:
    for t in a.tasks.split(","):
        _, test = M.load_split(t)
        for i, it in enumerate(test):
            if (t, i) in done:
                continue
            path, prompt, gold = M.parse_item(it)
            im = Image.open(path).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt + M.FORMAT[M.TASKS[t][2]]}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
            x = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                y = model.generate(**x, max_new_tokens=64, do_sample=False)
            out = proc.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({"task": t, "i": i, "pred": out, "gold": gold, "ok": bool(M.score(t, it, out))}) + "\n")
            f.flush()
        print(t, "done", flush=True)
