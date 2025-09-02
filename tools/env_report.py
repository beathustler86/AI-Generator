from __future__ import annotations
import importlib, json, platform, torch, sys
PKGS = ["torch","diffusers","transformers","huggingface_hub","xformers","bitsandbytes","flash_attn"]
def gather():
    out={}
    for n in PKGS:
        try:
            importlib.import_module("flash_attn._C" if n=="flash_attn" else n)
            base = n.split('.')[0]
            ver = getattr(importlib.import_module(base),"__version__","unknown")
            out[n]=ver
        except Exception as e:
            out[n]=f"missing({type(e).__name__})"
    gpu={"cuda":torch.cuda.is_available()}
    if gpu["cuda"]:
        try:
            gpu["name"]=torch.cuda.get_device_name(0)
            gpu["capability"]=".".join(map(str,torch.cuda.get_device_capability(0)))
            gpu["count"]=torch.cuda.device_count()
        except Exception:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "venv": sys.prefix != getattr(sys,"base_prefix",sys.prefix),
        "versions": out,
        "gpu": gpu
    }
if __name__ == "__main__":
    print(json.dumps(gather(), indent=2))
