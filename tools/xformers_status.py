import importlib, json, torch, time, sys
from pathlib import Path

# --- Ensure project root on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

out={}
spec = importlib.util.find_spec("xformers")
out["xformers_importable"]= bool(spec)
if spec:
    import xformers  # type: ignore
    out["xformers_version"]= getattr(xformers,"__version__","?")

# Optional: try loading pipeline if not already loaded
try:
    from src.modules import generation as gen
    if gen.current_model_target() and not gen.has_pipeline():
        # Attempt lightweight load (no image generation)
        gen.force_load_pipeline()
    pipe = gen._PIPELINE
    out["pipeline_loaded"]= bool(pipe is not None)
    if pipe and hasattr(pipe,"unet"):
        flag = getattr(pipe.unet,"_use_memory_efficient_attention_xformers", None)
        out["unet_memory_efficient_attention"]= flag
except Exception as e:
    out["pipeline_probe_error"]= str(e)

def attn_baseline():
    B,H,T,D = 1,8,4096,64
    q=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    k=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    v=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    torch.cuda.synchronize(); t=time.time()
    _ = torch.softmax((q @ k.transpose(-2,-1)) * (D**-0.5), dim=-1) @ v
    torch.cuda.synchronize()
    return (time.time()-t)*1000

times=[]
for _ in range(3): attn_baseline()
for _ in range(5): times.append(attn_baseline())
out["baseline_attn_ms_avg"]= round(sum(times)/len(times),2)

try:
    from xformers.ops import memory_efficient_attention
    B,H,T,D = 1,8,4096,64
    q=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    k=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    v=torch.randn(B,H,T,D,device="cuda",dtype=torch.float16)
    for _ in range(3): memory_efficient_attention(q,k,v)
    xs=[]
    for _ in range(5):
        torch.cuda.synchronize(); t=time.time()
        memory_efficient_attention(q,k,v)
        torch.cuda.synchronize(); xs.append((time.time()-t)*1000)
    out["xformers_attn_ms_avg"]= round(sum(xs)/len(xs),2)
    out["speedup_xformers_vs_baseline"]= round(out["baseline_attn_ms_avg"]/out["xformers_attn_ms_avg"],2)
except Exception as e:
    out["xformers_direct_op_error"]= str(e)

print(json.dumps(out, indent=2))
