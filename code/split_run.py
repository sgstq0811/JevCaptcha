"""Run a script with Clef-flash loaded in bf16 across two GPUs (same weights and dtype as the single-GPU run)."""
import os, sys, runpy, torch
from huggingface_hub import snapshot_download
path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, path)
import joint_schema_model as J
from pathlib import Path


def load_split(model_path, device="cuda", dtype=torch.bfloat16, **kw):
    import json
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    p = Path(model_path)
    mem = {(k if k == "cpu" else int(k)): v for k, v in (x.split(":") for x in os.environ.get("SPLIT_MEM", "0:10GiB,1:11GiB").split(","))}
    dm = "auto"
    if "cpu" in mem:                                   # overflow layers to CPU; embeddings and head stay on a GPU (the head reads their weights)
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig
        with init_empty_weights():
            em = Qwen3_5ForConditionalGeneration._from_config(AutoConfig.from_pretrained(p), torch_dtype=dtype)
        keep = int(os.environ.get("SPLIT_KEEP_GIB", "4"))   # room on GPU 1 for the embeddings moved there below
        m2 = dict(mem); m2[1] = f"{max(1, int(str(mem[1]).rstrip('GiB')) - keep)}GiB"
        dm = infer_auto_device_map(em, max_memory=m2, no_split_module_classes=em._no_split_modules, dtype=dtype)
        for k in list(dm):
            if "embed_tokens" in k or k.startswith("lm_head") or k.endswith("norm") and "layers" not in k:
                dm[k] = 1
    bb = Qwen3_5ForConditionalGeneration.from_pretrained(p, dtype=dtype, device_map=dm, **({} if "cpu" in mem else {"max_memory": mem}))
    bb.config.use_cache = False
    head = J.JointSchemaHead(**json.loads((p / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(p / "joint_head.safetensors"), strict=True)
    dev = bb.get_output_embeddings().weight.device
    head = head.to(device=dev, dtype=dtype)
    print("device map:", sorted(set(map(str, bb.hf_device_map.values()))), "head on", dev, flush=True)
    return J.ClefModel(bb, head).eval(), AutoProcessor.from_pretrained(p)


def forward(self, batch):
    base = self.language_model
    media = batch.get("media") or {}
    tm = base.model
    if not media and hasattr(tm, "language_model"):
        tm = tm.language_model
    out = tm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, return_dict=True, **media)
    w = base.get_output_embeddings().weight
    d = w.device
    return self.head(out.last_hidden_state.to(d), batch["input_ids"].to(d), batch["attention_mask"].to(d), batch["records"], w)


J.load_release_model = load_split
from transformers import AutoModelForImageTextToText
_fp = AutoModelForImageTextToText.from_pretrained.__func__


def fp_split(cls, *a, **kw):                     # generative backbones: same dtype, layers spread over the GPUs
    kw["device_map"] = "auto"
    kw["max_memory"] = {(k if k == "cpu" else int(k)): v for k, v in (x.split(":") for x in os.environ.get("SPLIT_MEM", "0:10GiB,1:11GiB").split(","))}
    return _fp(cls, *a, **kw)


AutoModelForImageTextToText.from_pretrained = classmethod(fp_split)
J.ClefModel.forward = forward
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
