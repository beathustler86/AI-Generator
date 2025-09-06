from __future__ import annotations
import os, time, hashlib, gc, shutil, threading
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple, Callable
import torch
from contextlib import suppress

# Base perf knobs
if torch.cuda.is_available():
    with suppress(Exception):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

try:
    import xformers  # type: ignore
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverMultistepScheduler as DPMSMS
)

def _log_event(rec: Dict[str, Any]):
    try:
        from src.modules.utils.telemetry import log_event
        log_event(rec)
    except Exception:
        pass

def _log_exception(e: Exception, context: str):
    try:
        from src.modules.utils.telemetry import log_exception
        log_exception(e, context=context)
    except Exception:
        pass

MODELS_ROOT = Path(os.environ.get(
    "MODELS_ROOT", r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models"
)).resolve()
TEXT_TO_IMAGE_DIR = MODELS_ROOT / "text_to_image"
TEXT_TO_VIDEO_DIR = MODELS_ROOT / "text_to_video"

PIPELINE_DISK_CACHE_DIR = Path(os.environ.get(
    "PIPELINE_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "pipeline_cache")
)).resolve()
PIPELINE_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# After PIPELINE_DISK_CACHE_DIR mkdir and before try: _DISK_CACHE_ENABLED
if os.getenv("PIPELINE_CACHE_DISABLE","0") == "1":
    try:
        _DISK_CACHE_ENABLED = False  # type: ignore
    except Exception:
        pass
try: _DISK_CACHE_ENABLED
except NameError: _DISK_CACHE_ENABLED = True
try: _CURRENT_TARGET
except NameError: _CURRENT_TARGET = None

_DISK_CACHE_MAX_ITEMS = int(os.environ.get("PIPELINE_CACHE_MAX_ITEMS","5"))
_DISK_CACHE_MAX_GB = float(os.environ.get("PIPELINE_CACHE_MAX_GB","50.0"))

_CURRENT_TARGET: Optional[str] = None
_PIPELINE = None
_PIPELINE_DEVICE: Optional[str] = None
_LOCK = threading.RLock()
LAST_SCHEDULER_NAME: Optional[str] = None
_LAST_VRAM_SNAPSHOT: Optional[Tuple[int,int,str]] = None

_SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "dpm-sde": DPMSolverSDEScheduler,
    "dpmpp_2m_karras": DPMSolverMultistepScheduler,
    "dpmpp-2m-karras": DPMSolverMultistepScheduler,
}

_PREV_STEP_AVG: Dict[str, float] = {}
_PERF_REG_THRESH = 1.35
_SDPA_FALLBACK_LOGGED = False
FREE_VRAM_AFTER_GEN = os.getenv("FREE_VRAM_AFTER_GEN","0") == "1"
_LAST_SDXL_DECODE_MS = 0
_SLOW_VAE_DETECTED = False
_VAE_FP32_FORCED = False
_SDXL_FORCE_TWO_PHASE = False

# ---------- SDXL flat / collapsed latent helpers ----------
def _sdxl_inflate_latent(lat: torch.Tensor) -> List[torch.Tensor]:
    try:
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        amp = float(os.getenv("SDXL_LATENT_RESCUE_AMP", "0.8"))
        scale_boost = float(os.getenv("SDXL_LATENT_SCALE_BOOST", "3.0"))
        return [
            lat + torch.randn_like(lat) * amp,
            lat + torch.randn_like(lat) * (amp * 0.5),
            lat * scale_boost,
            lat * scale_boost + torch.randn_like(lat) * (amp * 0.25),
        ]
    except Exception:
        return []

def _image_is_flat(pil_img, std_eps: float = 1e-3) -> bool:
    try:
        import numpy as np
        arr = np.asarray(pil_img).astype("float32") / 255.0
        return float(arr.std()) <= std_eps
    except Exception:
        return False

def _latent_std(t: torch.Tensor) -> float:
    try:
        return float(t.detach().float().std().item())
    except Exception:
        return -1.0

def _log_sdpa_fallback():
    global _SDPA_FALLBACK_LOGGED
    if _SDPA_FALLBACK_LOGGED: return
    _log_event({"event":"sdpa_fallback_used"}); _SDPA_FALLBACK_LOGGED=True

# >>> PATCH START: re-add memory optimization helper <<<
def _maybe_enable_memory_opts(pipe):
    t = time.time()
    enabled = []
    with suppress(Exception):
        if os.getenv("ENABLE_XFORMERS","1") == "1":
            if _XFORMERS_AVAILABLE and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
                enabled.append("xformers")
            else:
                _log_sdpa_fallback()
        if os.getenv("ENABLE_ATTN_SLICING","0") == "1" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            enabled.append("attn_slicing")
        if os.getenv("ENABLE_SEQUENTIAL_CPU_OFFLOAD","0") == "1" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            enabled.append("seq_offload")
    _log_event({
        "event":"model_load_phase",
        "phase":"memory_opts",
        "ms": int((time.time()-t)*1000),
        "enabled": enabled
    })
# >>> PATCH END <<<

def _enable_sdxl_vae_opts(pipe):
    try:
        vae = getattr(pipe, "vae", None)
        if not vae:
            return
        if os.getenv("SDXL_VAE_SLICING", "1") == "1" and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
            _log_event({"event": "sdxl_vae_slicing"})
        if os.getenv("SDXL_VAE_TILING", "0") == "1" and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
            _log_event({"event": "sdxl_vae_tiling"})
    except Exception as e:
        _log_exception(e, "sdxl_vae_opts")

def _sdxl_manual_decode_batch(pipe, latents: torch.Tensor, force_fp32: bool = False):
    from PIL import Image
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise RuntimeError("VAE missing for decode")
    want_stats = os.getenv("SDXL_DECODE_STATS", "0") == "1"

    # Fast path for Dreamshaper Turbo: single canonical divide + decode
    profile = _profile_for_target(current_model_target() or "")
    if profile == "turbo" and os.getenv("TURBO_SIMPLE_DECODE", "1") == "1":
        scale = float(getattr(vae.config, "scaling_factor", 0.18215))
        with torch.inference_mode():
            z = (latents.to(device=vae.device, dtype=vae.dtype)) / scale
            dec = _chunked_vae_decode_if_needed(vae, z)  # disables cudnn.benchmark during decode
            img = (dec/2+0.5).clamp(0,1)[0].detach().cpu()
            import numpy as np
            arr = (img.mul(255).round().byte().permute(1,2,0).numpy())
            return [Image.fromarray(arr)]

    base_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    alt_scales = [base_scale, base_scale * 1.5, base_scale * 2.0, base_scale * 0.5]
    imgs = []

    def stat(tag,t):
        if not want_stats: return
        with torch.no_grad():
            _log_event({"event":"sdxl_latent_stat", "tag":tag,
                        "mean":float(t.mean().item()),
                        "std":float(t.std().item()),
                        "min":float(t.min().item()),
                        "max":float(t.max().item())})
    stat("raw", latents)

    def build(scale:float, mode:str, cast_fp32:bool=False):
        t = latents.to(dtype=vae.dtype)
        if mode=="divide":
            t = t/scale
        elif mode=="none":
            pass
        elif mode=="multiply":
            t = t*scale
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    attempts = []
    attempts.append(("divide_base_fp16" if not force_fp32 else "divide_base_fp32",
                     lambda: build(base_scale,"divide",False)))
    for sc in alt_scales[1:]:
        attempts.append((f"divide_alt{sc:g}", lambda sc=sc: build(sc,"divide",False)))
    attempts.append(("no_divide", lambda: build(base_scale,"none",False)))
    attempts.append(("divide_base_force_fp32", lambda: build(base_scale,"divide",True)))
    attempts.append(("multiply_base", lambda: build(base_scale,"multiply",False)))
    for sc in alt_scales[1:]:
        attempts.append((f"multiply_alt{sc:g}", lambda sc=sc: build(sc,"multiply",False)))

    def decode_variant(name,tensor):
        try:
            with torch.inference_mode():
                tensor = tensor.to(device=vae.device, dtype=vae.dtype)
                # Always go through helper to keep cudnn.benchmark disabled during decode
                out = _chunked_vae_decode_if_needed(vae, tensor)
                if out is not None and torch.is_tensor(out) and float(out.float().std().item()) < 1e-6:
                    _log_event({"event": "sdxl_decode_zero_output", "variant": name})
                return out
        except Exception as e:
            _log_exception(e,f"sdxl_decode_{name}")
            return None

    def to_img(dec):
        im=(dec/2+0.5).clamp(0,1)[0].detach().cpu()
        return im

    flat_variants = 0
    chosen_variant = None
    for name,maker in attempts:
        lat_try=maker()
        stat(f"latent_{name}", lat_try)
        dec=decode_variant(name, lat_try)
        if dec is None:
            continue
        dec=torch.nan_to_num(dec, nan=0.0, posinf=0.0)
        img_t=to_img(dec)
        b=float(img_t.mean().item())
        s=float(img_t.std().item())
        _log_event({"event":"sdxl_decode_try","variant":name,"brightness":b,"std":s})
        flat = (s < 0.0008) or (b!=b)
        if flat:
            flat_variants += 1
        if not flat and chosen_variant is None:
            chosen_variant = img_t
            break

    if chosen_variant is None:
        if flat_variants == len(attempts):
            _log_event({"event":"sdxl_decode_all_flat","attempts":len(attempts)})
        try:
            rec_lat = (latents.float() if force_fp32 else latents.to(vae.dtype)) * (base_scale * 2.0)
            dec = _chunked_vae_decode_if_needed(vae, rec_lat)
            dec=torch.nan_to_num(dec, nan=0.0, posinf=0.0)
            rec_img=to_img(dec)
            rb=float(rec_img.mean().item()); rs=float(rec_img.std().item())
            _log_event({"event":"sdxl_decode_recover_attempt","brightness":rb,"std":rs})
            if rs >= 0.0008:
                chosen_variant = rec_img
        except Exception as e:
            _log_exception(e,"sdxl_decode_recover_attempt")

    if chosen_variant is None:
        try:
            lat_final=build(base_scale,"divide",True)
            dec=_chunked_vae_decode_if_needed(vae, lat_final)
            dec=torch.nan_to_num(dec, nan=0.0, posinf=0.0)
            chosen_variant=to_img(dec)
            b=float(chosen_variant.mean().item()); s=float(chosen_variant.std().item())
            _log_event({"event":"sdxl_decode_final_fallback","brightness":b,"std":s})
        except Exception:
            from PIL import Image
            imgs.append(Image.new("RGB",(latents.shape[-1]*8, latents.shape[-2]*8),(0,0,0)))
            _log_event({"event":"sdxl_decode_total_fail"})
            return imgs

    if chosen_variant is not None:
        import numpy as np
        arr=(chosen_variant.mul(255).round().byte().permute(1,2,0).numpy())
        imgs.append(Image.fromarray(arr))

    if not imgs:
        from PIL import Image
        imgs.append(Image.new("RGB",(latents.shape[-1]*8, latents.shape[-2]*8),(0,0,0)))
        _log_event({"event":"sdxl_decode_total_fail_post_recover"})
    return imgs

# --- Fast path helper (diagnostic) ---
def _sdxl_fast_decode_latent(pipe, latents: torch.Tensor):
    """
    Minimal decode (single attempt) for two-phase diagnostics.
      FAST_TWOPHASE_DECODE=1 to enable.
    """
    if latents.dim()==3:
        latents = latents.unsqueeze(0)
    latents = latents.detach().contiguous().clone()
    vae = getattr(pipe,"vae",None)
    if vae is None:
        raise RuntimeError("VAE missing for decode")
    scale = float(getattr(vae.config,"scaling_factor",0.18215))
    with torch.inference_mode():
        z = latents.to(device=vae.device, dtype=vae.dtype)
        # single canonical divide
        z = z / scale
        out = vae.decode(z).sample
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        img = (out/2+0.5).clamp(0,1)[0].detach().cpu()
        std_val = float(img.std().item())
        mean_val = float(img.mean().item())
        _log_event({"event":"sdxl_fast_decode_stat","mean":mean_val,"std":std_val})
        from PIL import Image
        import numpy as np
        arr=(img.mul(255).round().byte().permute(1,2,0).numpy())
        return Image.fromarray(arr), std_val


# --- Pipeline-native decode helper (preferred) ---
def _chunked_vae_decode_if_needed(vae, z: torch.Tensor):
    """
    Optionally split latent along height to reduce a very slow monolithic decode.
      VAE_DECODE_CHUNKS = N (>=1). If 1 => direct decode.
    Disables cudnn.benchmark during decode to avoid long per-call algo re-search.
    Emits vae_decode_chunked event.
    """
    chunks = int(os.getenv("VAE_DECODE_CHUNKS", "1"))
    prev_flag = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        if chunks <= 1:
            with torch.inference_mode():
                out = vae.decode(z).sample
            try:
                _log_event({"event": "vae_decode_direct_std",
                            "std": float(out.float().std().item())})
            except Exception:
                pass
            return out
        B,C,H,W = z.shape
        base = H // chunks
        rem = H % chunks
        h_cursor = 0
        parts = []
        for i in range(chunks):
            h = base + (1 if i < rem else 0)
            h_end = h_cursor + h
            slice_z = z[:, :, h_cursor:h_end, :]
            with torch.inference_mode():
                dec_part = vae.decode(slice_z).sample
            parts.append(dec_part)
            h_cursor = h_end
        out = torch.cat(parts, dim=2)
        try:
            _log_event({"event": "vae_decode_chunked",
                        "chunks": chunks,
                        "shape": [int(x) for x in out.shape],
                        "std": float(out.float().std().item())})
        except Exception:
            pass
        return out
    except Exception as e:
        _log_exception(e, "vae_decode_chunked_fail")
        with torch.inference_mode():
            out = vae.decode(z).sample
        return out
    finally:
        torch.backends.cudnn.benchmark = prev_flag

def _sdxl_pipeline_decode(pipe, latents: torch.Tensor):
    """
    Use pipeline.decode_latents if available (mirrors internal diffusers logic).
    Falls back to direct vae.decode if method missing.
    """
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    latents = latents.detach().contiguous().clone()
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise RuntimeError("VAE missing")
    scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    t0 = time.time()
    img_pil = None
    std_val = -1.0
    try:
        if hasattr(pipe, "decode_latents") and os.getenv("TWO_PHASE_USE_DECODE_LATENTS", "1") == "1":
            # Pass raw latents; decode_latents itself applies (1 / scaling_factor)
            z = latents.to(device=vae.device, dtype=vae.dtype)
            # Log pre-decode latent stats
            try:
                _log_event({
                    "event": "sdxl_pipeline_decode_input_stats",
                    "mean": float(z.float().mean().item()),
                    "std": float(z.float().std().item()),
                    "min": float(z.float().min().item()),
                    "max": float(z.float().max().item())
                })
            except Exception:
                pass
            decoded = pipe.decode_latents(z)
            # decode_latents returns np array [B,H,W,3] float in [0,1]
            if decoded is not None:
                import numpy as np
                arr = np.clip(decoded[0], 0, 1)
                std_val = float(arr.std())
                from PIL import Image
                img_pil = Image.fromarray((arr * 255).round().astype("uint8"))
                _log_event({"event": "sdxl_pipeline_decode_latents",
                            "ms": int((time.time()-t0)*1000),
                            "std": std_val})
                return img_pil, std_val
    except Exception as e:
        _log_exception(e, "pipeline_decode_latents")
    # Fallback: approximate internal logic (single manual division) with timing & optional forced fp16
    try:
        t_dec0 = time.time()
        _log_event({"event":"vae_decode_phase_start"})
        force_fp16_dec = os.getenv("VAE_FORCE_FP16_DECODE","0") == "1"
        if force_fp16_dec and vae.dtype != torch.float16:
            with suppress(Exception):
                vae_fp_orig = vae.dtype
                vae.to(torch.float16)
        z2 = (latents / scale).to(device=vae.device, dtype=vae.dtype if not force_fp16_dec else torch.float16)
        dec = _chunked_vae_decode_if_needed(vae, z2)
        dec_ms = int((time.time()-t_dec0)*1000)
        _log_event({"event":"vae_decode_timed","path":"fallback_direct","ms":dec_ms,
                    "dtype":str(vae.dtype),"force_fp16":force_fp16_dec})
        # Slow decode mitigation: attempt one-shot slicing or tiling if extremely slow
        if dec_ms > int(os.getenv("VAE_DECODE_SLOW_THRESH_MS","15000")) and os.getenv("VAE_SLOW_RETRY","1")=="1":
            try:
                _log_event({"event":"vae_decode_slow_detected","ms":dec_ms})
                # Enable slicing & retry quickly (lightweight)
                if hasattr(vae,"enable_slicing"):
                    vae.enable_slicing()
                    _log_event({"event":"vae_decode_retry_slicing_enabled"})
                t_r = time.time()
                with torch.inference_mode():
                    dec2 = vae.decode(z2).sample
                dec2_ms = int((time.time()-t_r)*1000)
                if float(dec2.float().std().item()) > 0 and dec2_ms < dec_ms:
                    dec = dec2
                    _log_event({"event":"vae_decode_retry_success","ms":dec2_ms})
                else:
                    _log_event({"event":"vae_decode_retry_no_improve","ms":dec2_ms})
            except Exception as e:
                _log_exception(e,"vae_decode_retry")
    except Exception as e:
        _log_exception(e, "pipeline_decode_fallback_direct")
    return img_pil, std_val

def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t

def _invalidate_prompt_cache():
    try:
        _PROMPT_EMBED_CACHE.clear()
        _PROMPT_CACHE_ORDER.clear()
        _log_event({"event":"prompt_cache_cleared"})
    except Exception:
        pass

def _is_diffusers_dir(p: Path) -> bool: return p.is_dir() and (p / "model_index.json").exists()

def list_local_image_models() -> List[str]:
    out=[]
    if TEXT_TO_IMAGE_DIR.is_dir():
        for d in TEXT_TO_IMAGE_DIR.iterdir():
            print(f"[DEBUG] Checking: {d}")  # Add this line
            if _is_diffusers_dir(d):
                print(f"[DEBUG] Detected model: {d}")  # Add this line
                out.append(str(d.resolve()))
            else:
                print(f"[DEBUG] Skipped (no model_index.json): {d}")  # Add this line
    for d in MODELS_ROOT.glob("models--*"):
        snap = d / "snapshots"
        if snap.is_dir():
            for s in snap.iterdir():
                if _is_diffusers_dir(s): out.append(str(s.resolve()))
    print(f"[DEBUG] Final detected models: {out}")  # Add this line
    return sorted(out)

def list_local_video_models() -> List[str]:
    out=[]
    if TEXT_TO_VIDEO_DIR.is_dir():
        for d in TEXT_TO_VIDEO_DIR.iterdir():
            if _is_diffusers_dir(d):
                out.append(str(d.resolve()))
    for d in MODELS_ROOT.iterdir():
        name=d.name.lower()
        if any(tag in name for tag in ("cosmos","comfy")) and _is_diffusers_dir(d):
            out.append(str(d.resolve()))
    if out:
        _log_event({"event":"video_models_found","count":len(out)})
    return sorted(set(out))

def list_all_models() -> List[Dict[str,Any]]:
    seen=set()
    items=[]
    def _friendly_hf_label(path_str: str) -> str:
        p = Path(path_str)
        parts = [x for x in p.parts]
        try:
            snap_idx = parts.index("snapshots")
            repo_node = parts[snap_idx-1]
            if repo_node.startswith("models--"):
                segs = repo_node.split("--")[1:]
                if len(segs) >= 2:
                    org = segs[0]; repo = "--".join(segs[1:])
                    label = f"{org}/{repo}"
                    if os.getenv("SHOW_REV_HASH","0")=="1":
                        label += f" ({parts[snap_idx+1][:7]})"
                    if label.lower()=="runwayml/stable-diffusion-v1-5":
                        label = "SD 1.5"
                    return label
        except ValueError:
            pass
        return Path(path_str).name

    for p in list_local_image_models():
        canon=os.path.normcase(os.path.abspath(p))
        if canon in seen: continue
        seen.add(canon)
        items.append({"label":_friendly_hf_label(p),"path":p,"kind":"image"})
    for p in list_local_video_models():
        canon=os.path.normcase(os.path.abspath(p))
        if canon in seen: continue
        seen.add(canon)
        items.append({"label":Path(p).name,"path":p,"kind":"video"})
    if not any("stable-diffusion-xl-base" in i["path"].lower() for i in items):
        items.append({"label":"HF: SDXL Base (hub)","path":"stabilityai/stable-diffusion-xl-base-1.0","kind":"image"})
    _log_event({"event":"model_catalog_built","image":len([i for i in items if i['kind']=='image']),
                "video":len([i for i in items if i['kind']=='video'])})
    return items

def _hash_key(text: str) -> str: return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
def _canonical(path: str) -> str:
    try: p = str(Path(path).resolve())
    except Exception: p = path
    if os.name == "nt": p = os.path.normcase(p)
    return p.replace("\\","/")

def _cache_slug_for(original: str) -> str:
    canon = _canonical(original); return f"{Path(canon).name[:48]}-{_hash_key(canon)}"
def _resolve_cached_path(original: str) -> Optional[str]:
    if not _DISK_CACHE_ENABLED: return None
    slug = _cache_slug_for(original)
    cdir = PIPELINE_DISK_CACHE_DIR / slug
    if (cdir/"model_index.json").exists():
        _log_event({"event":"disk_cache_hit","slug":slug})
        return str(cdir.resolve())
    return None

def _dir_size(p: Path) -> int:
    total=0
    for f in p.rglob("*"):
        if f.is_file():
            with suppress(Exception): total += f.stat().st_size
    return total

def _prune_disk_cache():
    if not PIPELINE_DISK_CACHE_DIR.exists(): return
    entries=[p for p in PIPELINE_DISK_CACHE_DIR.iterdir() if (p/"model_index.json").exists()]
    if not entries: return
    entries.sort(key=lambda p: p.stat().st_mtime)
    max_bytes=_DISK_CACHE_MAX_GB*(1024**3)
    total=sum(_dir_size(e) for e in entries)
    while (len(entries)>_DISK_CACHE_MAX_ITEMS or total>max_bytes) and entries:
        victim=entries.pop(0)
        freed=_dir_size(victim)
        shutil.rmtree(victim, ignore_errors=True)
        total-=freed
        _log_event({"event":"disk_cache_prune","path":str(victim),"freed_bytes":freed})

def _write_disk_pipeline_cache(original: str, pipe):
    if not _DISK_CACHE_ENABLED:
        _log_event({"event":"disk_cache_write_skip","reason":"disabled"}); return
    slug=_cache_slug_for(original)
    cdir=PIPELINE_DISK_CACHE_DIR/slug
    if cdir.exists():
        _log_event({"event":"disk_cache_write_skip","reason":"exists","slug":slug}); return
    with suppress(Exception):
        pipe.save_pretrained(str(cdir), safe_serialization=True)
        _log_event({"event":"disk_cache_write","slug":slug})
        _prune_disk_cache()

def is_disk_cache_enabled() -> bool: return bool(_DISK_CACHE_ENABLED)
def set_disk_cache_enabled(enabled: bool):
    global _DISK_CACHE_ENABLED
    _DISK_CACHE_ENABLED = bool(enabled)
    _log_event({"event":"disk_cache_set","enabled":_DISK_CACHE_ENABLED})
def get_cache_stats() -> Dict[str,Any]:
    entries=[]; total=0
    if PIPELINE_DISK_CACHE_DIR.exists():
        for d in PIPELINE_DISK_CACHE_DIR.iterdir():
            if (d/"model_index.json").exists():
                sz=_dir_size(d); total+=sz
                entries.append({"slug":d.name,"mb":int(sz/1024**2),"files":sum(1 for _ in d.rglob("*") if _.is_file())})
    return {"root":str(PIPELINE_DISK_CACHE_DIR),"enabled":_DISK_CACHE_ENABLED,"count":len(entries),"total_mb":int(total/1024**2),"entries":entries}

def purge_disk_cache():
    if PIPELINE_DISK_CACHE_DIR.exists():
        for d in PIPELINE_DISK_CACHE_DIR.iterdir():
            if d.is_dir(): shutil.rmtree(d, ignore_errors=True)
        _log_event({"event":"disk_cache_purged"})

def current_model_target() -> Optional[str]: return _CURRENT_TARGET
def set_model_target(target: str):
    global _CURRENT_TARGET
    if not target: return
    if _CURRENT_TARGET and _CURRENT_TARGET != target:
        release_pipeline(free_ram=False)
        _log_event({"event":"model_switch","from":_CURRENT_TARGET,"to":target})
        _invalidate_prompt_cache()
    _CURRENT_TARGET = target

def _vram_snapshot(phase: str|None=None, extra: Dict[str,Any]|None=None):
    if not torch.cuda.is_available(): return
    global _LAST_VRAM_SNAPSHOT
    try:
        torch.cuda.synchronize()
        alloc=int(torch.cuda.memory_allocated()//1024**2)
        reserved=int(torch.cuda.memory_reserved()//1024**2)
        key=_CURRENT_TARGET or "none"
        dedup=os.environ.get("VRAM_SNAPSHOT_DEDUP","1")!="0"
        changed=_LAST_VRAM_SNAPSHOT!=(alloc,reserved,key)
        if (not dedup) or phase or changed:
            rec={"event":"vram_snapshot","alloc_mb":alloc,"reserved_mb":reserved,
                 "target":_CURRENT_TARGET,"has_pipeline":_PIPELINE is not None}
            if phase: rec["phase"]=phase
            if extra: rec.update(extra)
            _log_event(rec)
            _LAST_VRAM_SNAPSHOT=(alloc,reserved,key)
    except Exception:
        pass

def _select_pipeline_class(target:str):
    return StableDiffusionXLPipeline if "xl" in target.lower() else StableDiffusionPipeline

def _apply_scheduler(pipe, name:str):
    global LAST_SCHEDULER_NAME
    if not name: name=os.environ.get("DEFAULT_SCHEDULER","")
    if not name or name==LAST_SCHEDULER_NAME: return
    cls=_SCHEDULER_MAP.get(name.lower())
    if not cls: return
    t0=time.time()
    try:
        cfg=pipe.scheduler.config
        if "dpmpp" in name.lower() and hasattr(cls,"from_config"):
            cfgd=dict(cfg); cfgd.update({"use_karras_sigmas":True,"algorithm_type":"dpmsolver++","solver_order":2,"timestep_spacing":"trailing"})
            pipe.scheduler=cls.from_config(cfgd)
        else:
            pipe.scheduler=cls.from_config(cfg)
        LAST_SCHEDULER_NAME=name
        _log_event({"event":"scheduler_set","name":name,"ms":int((time.time()-t0)*1000)})
    except Exception as e:
        _log_exception(e,"scheduler_set")

def validate_local_model(path:str)->Tuple[bool,List[str]]:
    p=Path(path); issues=[]
    if not p.exists(): issues.append("path_missing")
    if not (p/"model_index.json").exists(): issues.append("model_index.json missing")
    return (len(issues)==0, issues)

def _patch_vae_decode(pipe):
    vae = getattr(pipe, "vae", None)
    if not vae or getattr(vae, "_decode_patched", False):
        return
    orig = vae.decode
    ref_param = next(vae.parameters(), None)
    ref_dtype = ref_param.dtype if ref_param is not None else getattr(vae, "dtype", torch.float32)
    def _coerce(z):
        if torch.is_tensor(z):
            target_dtype = ref_dtype or vae.dtype
            if z.dtype != target_dtype:
                with suppress(Exception):
                    z = z.to(dtype=target_dtype, device=vae.device, non_blocking=True)
        return z
    prof = os.getenv("VAE_DECODE_PROF","0") == "1"
    def safe_decode(z, *a, **k):
        z = _coerce(z)
        try:
            t0 = time.time() if prof else None
            out = orig(z, *a, **k)
            if prof and t0 is not None:
                try:
                    _log_event({
                        "event":"vae_decode_call",
                        "ms": int((time.time()-t0)*1000),
                        "dtype": str(getattr(z, "dtype", None)),
                        "shape": list(getattr(z, "shape", []))
                    })
                except Exception:
                    pass
            return out
        except RuntimeError as e:
            msg = str(e)
            if ("Input type (float) and bias type" in msg or "input type (float)" in msg):
                try:
                    z = _coerce(z)
                    return orig(z, *a, **k)
                except Exception as e2:
                    try:
                        vae.float()
                        _log_event({"event":"sdxl_vae_auto_upcast_fp32"})
                        z2 = z.float() if torch.is_tensor(z) else z
                        return orig(z2, *a, **k)
                    except Exception as e3:
                        _log_exception(e3, "vae_decode_retry_fail")
                    _log_exception(e2, "vae_decode_retry_error")
            raise
    vae.decode = safe_decode  # type: ignore
    setattr(vae, "_decode_patched", True)
    _log_event({"event": "vae_decode_patched", "dtype": str(getattr(vae, 'dtype', None))})

def _self_test_vae(pipe, model_path: str):
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    try:
        with torch.inference_mode():
            lat = torch.randn(1, 4, 64, 64, device=vae.device, dtype=vae.dtype)
            dec = vae.decode(lat).sample
            base_std = float(dec.float().std().item())
        if base_std < 0.02:
            _log_event({"event": "sdxl_vae_selftest_fail", "std": base_std})
            try:
                vae.float()
                with torch.inference_mode():
                    lat2 = torch.randn(1, 4, 64, 64, device=vae.device, dtype=torch.float32)
                    dec2 = vae.decode(lat2).sample
                    std2 = float(dec2.float().std().item())
                if std2 >= 0.02:
                    _log_event({"event": "sdxl_vae_selftest_recover_fp32", "std": std2})
                    return
            except Exception as e:
                _log_exception(e, "vae_selftest_upcast")
            try:
                from diffusers import AutoencoderKL
                new_vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
                new_vae.to(vae.device)
                pipe.vae = new_vae
                with torch.inference_mode():
                    lat3 = torch.randn(1, 4, 64, 64, device=new_vae.device, dtype=new_vae.dtype)
                    dec3 = new_vae.decode(lat3).sample
                    std3 = float(dec3.float().std().item())
                if std3 < 0.02:
                    os.environ["DISABLE_SDXL_MANUAL_DECODE"] = "1"
                    os.environ["SDXL_TWO_PHASE"] = "0"
                    _log_event({"event": "sdxl_manual_decode_disabled_fallback", "std": std3})
                else:
                    _log_event({"event": "sdxl_vae_reloaded", "std": std3})
                return
            except Exception as e:
                _log_exception(e, "vae_reload_fail")
                os.environ["DISABLE_SDXL_MANUAL_DECODE"] = "1"
                os.environ["SDXL_TWO_PHASE"] = "0"
                _log_event({"event": "sdxl_manual_decode_disabled_fallback"})
    except Exception as e:
        _log_exception(e, "vae_selftest")

def _align_pipeline_dtypes(pipe):
    """
    Safe alignment:
      - Default: allow mixed precision (VAE fp32, UNet fp16) to save VRAM.
      - Set DTYPE_ALIGN_STRICT=1 to force UNet to match VAE.
      - Set DTYPE_ALIGN_MATCH_UNET=1 to downcast VAE to UNet dtype instead.
      - Auto-downgrade UNet if VRAM usage exceeds UNET_DOWNGRADE_THRESH (fraction).
    """
    if os.getenv("ENABLE_DTYPE_ALIGN", "1") != "1":
        return
    vae = getattr(pipe, "vae", None)
    unet = getattr(pipe, "unet", None)
    if not vae or not unet:
        return
    strict = os.getenv("DTYPE_ALIGN_STRICT", "0") == "1"
    match_unet = os.getenv("DTYPE_ALIGN_MATCH_UNET", "0") == "1"
    allow_mixed = not strict and not match_unet
    changed = False
    action = "noop"
    try:
        if allow_mixed:
            # Mixed is acceptable; do nothing unless VAE lower precision than UNet (rare)
            if vae.dtype == torch.float16 and unet.dtype == torch.float32:
                with suppress(Exception):
                    vae.to(unet.dtype)
                    changed = True
                    action = "vae_upcast_to_unet"
            else:
                action = "mixed_ok"
        elif strict:
            if unet.dtype != vae.dtype:
                with suppress(Exception):
                    unet.to(vae.dtype)
                    changed = True
                    action = f"unet_to_{vae.dtype}"
        elif match_unet:
            if vae.dtype != unet.dtype:
                with suppress(Exception):
                    vae.to(unet.dtype)
                    changed = True
                    action = f"vae_to_{unet.dtype}"
    except Exception as e:
        _log_exception(e, "dtype_align")
        action = "error"
    _log_event({
        "event": "dtype_align",
        "vae": str(getattr(vae, "dtype", None)),
        "unet": str(getattr(unet, "dtype", None)),
        "changed": changed,
        "mode": ("strict" if strict else "match_unet" if match_unet else "mixed"),
        "action": action
    })
    _maybe_autodowngrade_unet(pipe)

def _maybe_autodowngrade_unet(pipe):
    if os.getenv("UNET_AUTODOWNGRADE", "1") != "1":
        return
    if not torch.cuda.is_available():
        return
    thresh = float(os.getenv("UNET_DOWNGRADE_THRESH", "0.88"))
    try:
        unet = getattr(pipe, "unet", None)
        if not unet:
            return
        if unet.dtype == torch.float16:
            return
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        usage = reserved / total
        if usage >= thresh:
            with suppress(Exception):
                unet.to(torch.float16)
                _log_event({
                    "event": "unet_autodowngrade_fp16",
                    "usage_pct": round(usage * 100, 2)
                })
    except Exception as e:
        _log_exception(e, "unet_autodowngrade")

# --- Memory format + optional compile helpers (re-added) ---

def _apply_memory_format(pipe):
    """
    Set memory_format for major submodules:
      - UNet: channels_last is usually faster on CUDA.
      - VAE: prefer contiguous on SDXL/Turbo (stable on Windows).
        Set DISABLE_VAE_CHANNELS_LAST=0 to allow channels_last for non-SDXL.
    """
    t = time.time()
    try:
        with suppress(Exception):
            if hasattr(pipe, "unet"):
                pipe.unet.to(memory_format=torch.channels_last)
            if hasattr(pipe, "vae"):
                disable_vae_cl = os.getenv("DISABLE_VAE_CHANNELS_LAST", "1") == "1"
                is_xl = "xl" in (current_model_target() or "").lower()
                if disable_vae_cl or is_xl:
                    pipe.vae.to(memory_format=torch.contiguous_format)
                else:
                    pipe.vae.to(memory_format=torch.channels_last)
    finally:
        _log_event({
            "event": "model_load_phase",
            "phase": "memory_format",
            "ms": int((time.time()-t)*1000)
        })

_COMPILED = False
import platform

def _maybe_compile(pipe):
    """
    Optionally compile UNet with torch.compile when ENABLE_TORCH_COMPILE=1.
    On Windows, requires triton; otherwise we skip safely.
    """
    global _COMPILED
    if _COMPILED or os.getenv("ENABLE_TORCH_COMPILE","0") != "1":
        return
    if platform.system() == "Windows":
        try:
            import triton  # noqa: F401
        except Exception:
            _log_event({"event":"torch_compile_skip","reason":"no_triton_windows"})
            return
    t = time.time()
    try:
        pipe.unet = torch.compile(
            pipe.unet,
            mode="reduce-overhead",
            fullgraph=False,
            backend="inductor"
        )
        _COMPILED = True
        _log_event({"event":"torch_compile_unet","backend":"inductor","ms": int((time.time()-t)*1000)})
    except Exception as e:
        _log_exception(e, "torch_compile_unet")
        _log_event({"event":"torch_compile_skip","reason":type(e).__name__})

_COMPILED_VAE = False

def _maybe_compile_vae(pipe):
    """
    Optionally compile VAE with torch.compile when ENABLE_TORCH_COMPILE_VAE=1.
    On Windows, requires triton; otherwise we skip safely.
    """
    global _COMPILED_VAE
    if _COMPILED_VAE or os.getenv("ENABLE_TORCH_COMPILE_VAE","0") != "1":
        return
    if platform.system() == "Windows":
        try:
            import triton  # noqa: F401
        except Exception:
            _log_event({"event":"torch_compile_vae_skip","reason":"no_triton_windows"})
            return
    t = time.time()
    try:
        pipe.vae = torch.compile(
            pipe.vae,
            mode="reduce-overhead",
            fullgraph=False,
            backend="inductor"
        )
        _COMPILED_VAE = True
        _log_event({"event":"torch_compile_vae","backend":"inductor","ms": int((time.time()-t)*1000)})
    except Exception as e:
        _log_exception(e, "torch_compile_vae")
        _log_event({"event":"torch_compile_vae_skip","reason":type(e).__name__})

def _load_pipeline(device:str, sampler:str):
    global _PIPELINE,_PIPELINE_DEVICE
    with _LOCK:
        if _PIPELINE and _PIPELINE_DEVICE==device:
            _apply_scheduler(_PIPELINE, sampler)
            return _PIPELINE
        if not _CURRENT_TARGET: raise RuntimeError("No model target set.")
        tgt=_CURRENT_TARGET
        is_local= not ("/" in tgt and not Path(tgt).exists())
        if is_local:
            ok,issues=validate_local_model(tgt)
            if not ok:
                _log_event({"event":"model_validation_failed","target":tgt,"issues":issues})
                raise RuntimeError(f"Invalid model dir: {issues}")
        cached=_resolve_cached_path(tgt) if is_local else None
        load_path=cached or tgt
        if is_local and not ok:
            _log_event({"event":"disk_cache_write_skip","reason":"model_invalid","issues":issues,"slug":_cache_slug_for(tgt)})
        use_cuda=(device=="cuda" and torch.cuda.is_available())
        dtype=torch.float16 if use_cuda else torch.float32
        pipe_cls=_select_pipeline_class(load_path)
        t_from=time.time()
        pipe=pipe_cls.from_pretrained(load_path,
                                      torch_dtype=dtype,
                                      use_safetensors=True,
                                      low_cpu_mem_usage=True,
                                      safety_checker=None,
                                      local_files_only=False)

        # --- Model-specific defaults right after load ---
        if "turbo" in tgt.lower():
            # Dreamshaper XL V2 Turbo
            try:
                pipe.unet.to(torch.float16)
                force_vae_fp32 = os.getenv("TURBO_FORCE_VAE_FP32", "1") == "1"
                if force_vae_fp32:
                    pipe.vae.float()
                else:
                    pipe.vae.to(torch.float16)
                pipe.vae.to(memory_format=torch.contiguous_format)
                if hasattr(pipe.vae, "disable_slicing"):
                    pipe.vae.disable_slicing()
                    _log_event({"event": "vae_slicing_disabled", "reason": "turbo"})
                if hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
                    _log_event({"event": "vae_tiling_disabled", "reason": "turbo"})
                print(f"[Dreamshaper Turbo] VAE {'FP32' if force_vae_fp32 else 'FP16'}; slicing/tiling disabled.")
                if _XFORMERS_AVAILABLE and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[Dreamshaper Turbo] Setup failed: {e}")
        elif isinstance(pipe, StableDiffusionXLPipeline):
            # SDXL 1.0 base: FP32 VAE; slicing/tiling will be decided per-shape at generation time
            try:
                pipe.vae.float()
                pipe.vae.to(memory_format=torch.contiguous_format)
                # Avoid forcing slicing on load; generation step will toggle as needed.
                if hasattr(pipe.vae, "disable_slicing"):
                    pipe.vae.disable_slicing()
                    _log_event({"event":"sdxl_vae_slicing_default_off"})
                if hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
                print("[SDXL] VAE FP32; slicing decided per image size.")
                _log_event({"event":"sdxl_vae_fp32_forced"})
            except Exception as e:
                print(f"[SDXL] VAE FP32/contiguous failed: {e}")
        # ------------------------------------------------

        _log_event({"event":"model_load_phase","phase":"from_pretrained","ms":int((time.time()-t_from)*1000),"cached":bool(cached)})
        _maybe_enable_memory_opts(pipe)
        _apply_memory_format(pipe)

        _align_pipeline_dtypes(pipe)
        t_dtype=time.time()
        if use_cuda and _should_use_bf16(pipe_cls, tgt):
            with suppress(Exception):
                pipe.to(torch.bfloat16)
                os.environ["ENABLE_DTYPE_ALIGN"] = "0"
                _log_event({"event":"pipeline_dtype","dtype":"bfloat16"})
                if hasattr(pipe,"vae"):
                    pipe.vae.to(dtype=torch.float16); _log_event({"event":"vae_forced_fp16"})
        elif (use_cuda and hasattr(pipe,"vae")
              and os.getenv("FORCE_FP16_VAE","1")=="1"
              and os.getenv("DISABLE_FP16_VAE","0")!="1"
              and "turbo" not in tgt.lower()
              and not isinstance(pipe, StableDiffusionXLPipeline)):
            with suppress(Exception):
                pipe.vae.to(torch.float16); _log_event({"event":"vae_fp16_enforced"})
        _log_event({"event":"model_load_phase","phase":"dtype_adjust","ms":int((time.time()-t_dtype)*1000)})

        t_move=time.time()
        if use_cuda: pipe.to("cuda")
        _log_event({"event":"model_load_phase","phase":"move_to_cuda","ms":int((time.time()-t_move)*1000)})
        _apply_scheduler(pipe, sampler)
        _PIPELINE,_PIPELINE_DEVICE=pipe,device
        if is_local and not cached and _DISK_CACHE_ENABLED: _write_disk_pipeline_cache(tgt, pipe)
        _vram_snapshot("post_load", {"model":load_path})
        _maybe_compile(pipe)
        _maybe_compile_vae(pipe)

        # Force DPM++ 2M Karras for SDXL base only (keep user's sampler for Turbo)
        with suppress(Exception):
            if isinstance(pipe, StableDiffusionXLPipeline) and "turbo" not in tgt.lower():
                cfg=dict(pipe.scheduler.config)
                cfg.update({"use_karras_sigmas":True,"algorithm_type":"dpmsolver++","solver_order":2,"timestep_spacing":"trailing"})
                pipe.scheduler=DPMSMS.from_config(cfg)
                _log_event({"event":"scheduler_set_forced","name":"dpmpp_2m_karras"})

        # Do not auto-enable VAE slicing here; handled per image in generate_images
        _patch_vae_decode(pipe)
        _align_pipeline_dtypes(pipe)
        _self_test_vae(pipe, load_path)

        if os.getenv("WARMUP_ENABLE","1")=="1" and not cached:
            t_w=time.time()
            with suppress(Exception):
                with torch.inference_mode():
                    _=pipe(prompt="warmup", negative_prompt=None,num_inference_steps=2,guidance_scale=0.0,width=512,height=512).images
                _log_event({"event":"warmup_done","ms":int((time.time()-t_w)*1000)})
        else:
            _log_event({"event":"warmup_skipped","reason":("cached" if cached else "disabled")})

        ms_total=int((time.time()-t_from)*1000)
        _log_event({"event":"model_load_time","model":load_path,"ms":ms_total,"cached":bool(cached)})
        return _PIPELINE

def release_pipeline(free_ram: bool=False):
    global _PIPELINE,_PIPELINE_DEVICE
    with _LOCK:
        if not _PIPELINE: return
        with suppress(Exception):
            with torch.no_grad(): _PIPELINE.to("cpu")
        if free_ram: _PIPELINE=None
        _PIPELINE_DEVICE=None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        _vram_snapshot("release", {"free_ram":free_ram})

def force_load_pipeline(sampler:str="euler_a", device:str="cuda")->bool:
    if not _CURRENT_TARGET: raise RuntimeError("No model target set.")
    try: _load_pipeline(device=device, sampler=sampler); return True
    except Exception as e:
        _log_exception(e,"force_load_pipeline"); return False

_PROMPT_EMBED_CACHE: Dict[str, Dict[str, Any]] = {}
_PROMPT_CACHE_ORDER: List[str] = []
def _cache_key(prompt:str, neg:str, cfg:float, w:int,h:int, device:str)->str:
    return hashlib.sha256(((current_model_target() or "")+"\n"+prompt+"\n"+neg+f"\n{cfg}\n{w}x{h}\n{device}").encode("utf-8")).hexdigest()
def _get_prompt_embeds(pipe, prompt:str, negative:str, cfg:float, w:int,h:int, device:str):
    if os.getenv("ENABLE_PROMPT_CACHE","1")!="1": return None
    encode_fn=getattr(pipe,"_encode_prompt",None)
    if encode_fn is None: return None

    key=_cache_key(prompt, negative, cfg, w, h, device)
    rec=_PROMPT_EMBED_CACHE.get(key)
    if rec:
        _log_event({"event":"prompt_cache_hit"}); return rec["pos"],rec["neg"]

    # Chunking setup
    use_chunking = (os.getenv("ENABLE_PROMPT_CHUNKING","1") == "1")
    tokenizer = getattr(pipe, "tokenizer", None)
    # SDXL has two tokenizers; we use the primary for length decisions
    if tokenizer is None:
        with suppress(Exception):
            tokenizer = getattr(pipe, "tokenizer_2", None)

    # Determine effective token limit (reserve room for special tokens)
    eff_max = None
    if tokenizer is not None:
        with suppress(Exception):
            eff_max = int(getattr(tokenizer, "model_max_length", 77) or 77)
    # Allow env override
    with suppress(Exception):
        ov = int(os.getenv("CHUNK_MAX_TOKENS","0") or "0")
        if ov > 0:
            eff_max = ov
    if eff_max is None:
        eff_max = 77
    eff_body = max(8, eff_max - 2)  # reserve a couple for special tokens

    def _within(text: str) -> bool:
        try:
            if tokenizer is None:
                return len((text or "").split()) <= eff_body
            # Prefer excluding special tokens
            return len(tokenizer.encode(text or "", add_special_tokens=False)) <= eff_body
        except TypeError:
            return len(tokenizer.encode(text or "")) <= eff_body
        except Exception:
            return len((text or "").split()) <= eff_body

    # If no chunking or both within limit, do normal encode
    if (not use_chunking) or (_within(prompt) and _within(negative or "")):
        try:
            pos,neg = encode_fn(
                prompt=prompt,
                negative_prompt=negative or None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            if torch.is_tensor(pos) and pos.dim()==2:
                pos=_ensure_3d(pos); _log_event({"event":"prompt_cache_shape_fix","which":"pos"})
            if torch.is_tensor(neg) and neg.dim()==2:
                neg=_ensure_3d(neg); _log_event({"event":"prompt_cache_shape_fix","which":"neg"})
            _PROMPT_EMBED_CACHE[key]={"pos":pos,"neg":neg,"ts":time.time()}
            _PROMPT_CACHE_ORDER.append(key)
            max_items=int(os.getenv("MAX_PROMPT_CACHE","64"))
            if len(_PROMPT_CACHE_ORDER)>max_items:
                old=_PROMPT_CACHE_ORDER.pop(0); _PROMPT_EMBED_CACHE.pop(old, None)
            _log_event({"event":"prompt_cache_store","size":len(_PROMPT_CACHE_ORDER)})
            return pos, neg
        except Exception as e:
            _log_exception(e,"prompt_cache_encode")
            return None

    # Chunked path
    try:
        from src.modules.chunking_module import chunk_text
    except Exception as e:
        _log_exception(e, "prompt_chunk_import")
        return None

    # Build chunks
    pos_chunks = chunk_text(prompt or "", max_tokens=eff_body, tokenizer=tokenizer)
    neg_needs_chunk = bool(negative) and (not _within(negative))
    neg_chunks = chunk_text(negative or "", max_tokens=eff_body, tokenizer=tokenizer) if neg_needs_chunk else None

    # Helper: weigh a chunk by its token count
    def _weight_for(text: str) -> float:
        try:
            if tokenizer is None:
                return float(max(1, len((text or "").split())))
            try:
                return float(len(tokenizer.encode(text or "", add_special_tokens=False)))
            except TypeError:
                return float(len(tokenizer.encode(text or "")))
        except Exception:
            return 1.0

    pos_list=[]; neg_list=[]; w_list=[]
    for i, cp in enumerate(pos_chunks):
        cn = None
        if negative:
            if neg_chunks is not None:
                cn = neg_chunks[min(i, len(neg_chunks)-1)]
            else:
                cn = negative
        try:
            p_i, n_i = encode_fn(
                prompt=cp,
                negative_prompt=cn or None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            if torch.is_tensor(p_i) and p_i.dim()==2: p_i=_ensure_3d(p_i)
            if torch.is_tensor(n_i) and n_i.dim()==2: n_i=_ensure_3d(n_i)
            pos_list.append(p_i); neg_list.append(n_i); w_list.append(_weight_for(cp))
        except Exception as e:
            _log_exception(e, "prompt_chunk_encode")
            continue

    if not pos_list or not neg_list:
        _log_event({"event":"prompt_chunking_fallback_nochunks"})
        return None

    # Weighted average across chunks (preserve original dtype)
    w = torch.tensor(w_list, dtype=torch.float32, device=pos_list[0].device)
    w = w / (w.sum() if float(w.sum().item()) > 0 else 1.0)

    def _combine(tensors: List[torch.Tensor]) -> torch.Tensor:
        dtype = tensors[0].dtype
        stacked = torch.stack([t.float() for t in tensors], dim=0)  # [chunks,B,seq,dim]
        w_exp = w.view(-1, *([1] * (stacked.dim() - 1)))
        out = (stacked * w_exp).sum(dim=0).to(dtype=dtype)
        return out

    pos_comb = _combine(pos_list)
    neg_comb = _combine(neg_list)

    _PROMPT_EMBED_CACHE[key] = {"pos": pos_comb, "neg": neg_comb, "ts": time.time()}
    _PROMPT_CACHE_ORDER.append(key)
    max_items = int(os.getenv("MAX_PROMPT_CACHE", "64"))
    if len(_PROMPT_CACHE_ORDER) > max_items:
        old = _PROMPT_CACHE_ORDER.pop(0)
        _PROMPT_EMBED_CACHE.pop(old, None)

    _log_event({
        "event": "prompt_chunking_used",
        "pos_chunks": len(pos_chunks),
        "neg_chunks": (len(neg_chunks) if neg_chunks is not None else 1),
        "token_limit": eff_body
    })
    return pos_comb, neg_comb

# --- People/anatomy helpers (place near other helpers) ---
def _is_people_prompt(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    tokens = (
        "person","people","human","man","woman","boy","girl","male","female",
        "portrait","face","model","actor","actress","body","full body","selfie","hands","fingers","arm","leg"
    )
    return any(tok in t for tok in tokens)

def _augment_anatomy_prompt(prompt: str) -> str:
    base_aug = os.getenv(
        "ANATOMY_GUARD_POSITIVE_AUG_TEXT",
        "one person, full body, 2 arms, 2 legs, 5 fingers per hand, realistic hands, coherent anatomy, symmetrical body"
    )
    p = prompt or ""
    keys = ("2 arms","2 legs","5 fingers","coherent anatomy","realistic hands","symmetrical")
    if any(k.lower() in p.lower() for k in keys):
        return p
    joiner = (", " if p and not p.strip().endswith(",") else "")
    return f"{p}{joiner}{base_aug}"

def _apply_anatomy_guard(
    profile: str,
    sampler: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    width: int,
    height: int
) -> Tuple[str, str, int, float, int, int, str, Dict[str, Any]]:
    """
    People-aware overrides for better anatomy stability, opt-in via env:
      ANATOMY_GUARD=1 (default 1), ANATOMY_GUARD_SAMPLER (default dpmpp_2m_karras),
      ANATOMY_GUARD_GUIDANCE_RESCALE (default 0.7),
      ANATOMY_GUARD_POSITIVE_AUG (default 1), ANATOMY_GUARD_AUTO_PORTRAIT (default 0),
      ANATOMY_GUARD_TURBO_CFG_MAX (default 5.5), ANATOMY_GUARD_TURBO_STEPS (default 10).
    """
    if os.getenv("ANATOMY_GUARD", "1") != "1":
        return prompt, negative, steps, cfg, width, height

    is_people = _is_people_prompt(prompt) or _is_people_prompt(negative)
    if not is_people:
        return prompt, negative, steps, cfg, width, height

    changed = False
    meta: Dict[str, Any] = {"profile": profile, "applied": True}

    if os.getenv("ANATOMY_GUARD_AUTO_PORTRAIT", "0") == "1" and width > height:
        width, height = height, width
        changed = True
        meta["auto_portrait"] = True

    if os.getenv("ANATOMY_GUARD_POSITIVE_AUG", "1") == "1":
        new_prompt = _augment_anatomy_prompt(prompt)
        if new_prompt != prompt:
            prompt = new_prompt
            changed = True
            meta["positive_aug"] = True

    want_sampler = os.getenv("ANATOMY_GUARD_SAMPLER", "dpmpp_2m_karras").strip() or sampler
    if want_sampler and want_sampler.lower() != (sampler or "").lower():
        sampler = want_sampler
        changed = True
        meta["sampler"] = sampler

    if profile == "turbo":
        cfg_max = float(os.getenv("ANATOMY_GUARD_TURBO_CFG_MAX", "5.5"))
        if cfg > cfg_max:
            cfg = cfg_max
            changed = True
            meta["cfg_clamped"] = cfg
        min_steps = int(os.getenv("ANATOMY_GUARD_TURBO_STEPS", "10"))
        if steps < min_steps:
            steps = min_steps
            changed = True
            meta["steps_bumped"] = steps

    grescale = float(os.getenv("ANATOMY_GUARD_GUIDANCE_RESCALE", "0.7"))
    meta["guidance_rescale"] = max(0.0, min(1.0, grescale))

    if changed:
        _log_event({"event": "anatomy_guard_applied", **meta})
    else:
        _log_event({"event": "anatomy_guard_detected_people_nochange", **meta})
    return prompt, negative, steps, cfg, width, height, sampler, meta

# --- Generation toggles / helpers needed by loader and run ---
def _should_use_bf16(pipe_cls: Any, target: str) -> bool:
    if os.getenv("ENABLE_BF16", "1") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    supported = False
    try:
        supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass
    if not supported:
        try:
            major, _ = torch.cuda.get_device_capability()
            supported = (major >= 8)  # Ampere/Ada+
        except Exception:
            supported = False
    if not supported:
        _log_event({"event": "bf16_skip", "reason": "unsupported_device"})
        return False
    if "turbo" in (target or "").lower():
        _log_event({"event": "bf16_skip", "reason": "turbo_target"})
        return False
    return True

def _profile_for_target(path: str) -> str:
    p = (path or "").lower()
    if "turbo" in p:
        return "turbo"
    if "refiner" in p:
        return "refiner"
    if "xl" in p:
        return "sdxl"
    return "sd1"

def _adapt_steps(user_steps: int, profile: str, use_refiner: bool) -> int:
    # Turbo benefits from very low steps; SDXL/SD1 follow requested steps
    if profile == "turbo":
        cap = int(os.getenv("TURBO_STEPS_CAP", "8"))
        return min(user_steps, cap)
    return user_steps

# Ensure generation run counter exists
try:
    _GENERATION_RUN_COUNTER  # noqa: F821
except NameError:
    _GENERATION_RUN_COUNTER = 0

def _set_vae_slicing_for_shape(pipe, width: int, height: int, profile: str):
    """
    Enable/disable VAE slicing dynamically based on resolution to avoid slow decodes.
    Safe no-op if methods not present. Turbo keeps slicing disabled for speed.
    """
    try:
        vae = getattr(pipe, "vae", None)
        if not vae:
            return
        if profile == "turbo":
            return
        pixels = max(1, int(width) * int(height))
        px_thr = int(os.getenv("VAE_SLICING_PX_THRESHOLD", "1105920"))  # ~1080p
        if pixels >= px_thr:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
        else:
            if hasattr(vae, "disable_slicing"):
                vae.disable_slicing()
    except Exception as e:
        _log_exception(e, "set_vae_slicing_for_shape")

def _recover_black_pipeline_image(pipe, prompt: str, negative: str, gen_kwargs: dict, profile: str):
    """
    Fallback when pipeline returns empty images (rare). Re-run once to latent and decode.
    Returns PIL.Image or None.
    """
    try:
        if profile != "sdxl":
            return None
        args = dict(gen_kwargs)
        args.pop("prompt_embeds", None)
        args.pop("negative_prompt_embeds", None)
        args.update(prompt=prompt, negative_prompt=negative or None, output_type="latent")
        out = pipe(**args)
        latents = getattr(out, "latents", None) or getattr(out, "images", None)
        if latents is None:
            return None
        img, std_val = _sdxl_pipeline_decode(pipe, latents)
        if img is not None and std_val >= 0.0008:
            return img
        imgs = _sdxl_manual_decode_batch(pipe, latents, force_fp32=False)
        return imgs[0] if imgs else None
    except Exception as e:
        _log_exception(e, "recover_black_pipeline_image")
        return None

# --- Main generation entry point (called by UI) ---
def generate_images(
    prompt:str,
    negative:str,
    steps:int,
    cfg:float,
    width:int,
    height:int,
    seed:int,
    batch:int,
    sampler:str,
    device:str="cuda",
    progress_cb:Optional[Callable[[str],None]]=None,
    cancel_event:Optional[threading.Event]=None,
    _internal_retry:bool=False
)->List[Any]:
    manual_decode = (os.getenv("MANUAL_DECODE","1")!="0")
    t_start=time.time()
    tgt = current_model_target()
    if tgt:
        ok, issues = validate_local_model(tgt)
        print(f"[DEBUG] Model validation: ok={ok}, issues={issues}")
    pipe=_load_pipeline(device=device, sampler=sampler)

    global _GENERATION_RUN_COUNTER
    _GENERATION_RUN_COUNTER += 1
    run_id=_GENERATION_RUN_COUNTER
    _log_event({"event":"generation_run_start","run_id":run_id,"target":current_model_target()})

    try:
        _log_event({"event":"anomaly_rerun_counter_init",
                    "value": int(os.getenv("_ANOMALY_RERUNS_DONE","0"))})
    except Exception:
        pass

    prompt=prompt.strip()
    negative=(negative or "").strip()

    profile=_profile_for_target(current_model_target() or "")
    if profile=="turbo":
        manual_decode = True
        _log_event({"event":"manual_decode_forced","reason":"turbo_no_images"})

    if profile=="turbo" and manual_decode and os.getenv("MANUAL_DECODE_SKIP_TURBO","0")=="1":
        manual_decode=False
        _log_event({"event":"manual_decode_skipped","reason":"turbo_forced"})
    if profile=="sdxl" and manual_decode and os.getenv("DISABLE_SDXL_MANUAL_DECODE","1")=="1":
        manual_decode=False
        _log_event({"event":"manual_decode_disabled","reason":"sdxl_default"})
    if (profile=="sd1" and os.getenv("SDXL_STRICT","0")=="1"
        and any(tag in prompt.lower() for tag in ("sdxl","xl-base","8k","photoreal anime"))):
        _log_event({"event":"profile_mismatch_abort","profile":profile,"len":len(prompt)})
        raise RuntimeError("Prompt appears tailored for SDXL but SD1 model is active. Switch to SDXL or disable SDXL_STRICT.")

    use_refiner_env=os.getenv("USE_REFINER","0")=="1"
    adapt_env = os.getenv("ADAPT_STEPS","0")=="1" or (profile=="turbo" and os.getenv("ADAPT_STEPS") is None)
    orig_steps=steps
    if adapt_env:
        steps=_adapt_steps(steps, profile, use_refiner_env)
        if steps!=orig_steps: _log_event({"event":"steps_adapted","from":orig_steps,"to":steps,"profile":profile})

    # Anatomy Guard: may alter prompt/negative/steps/cfg/sampler/size.
    anat_meta: Dict[str, Any] = {}
    cache_invalidate = False
    try:
        prompt2, negative2, steps2, cfg2, width2, height2, sampler2, anat_meta = _apply_anatomy_guard(
            profile, sampler, prompt, negative, steps, cfg, width, height
        )
        if sampler2 != sampler:
            _apply_scheduler(pipe, sampler2)
            sampler = sampler2
        cache_invalidate = (prompt2 != prompt) or (negative2 != negative) or (cfg2 != cfg)
        prompt, negative, steps, cfg, width, height = prompt2, negative2, steps2, cfg2, width2, height2
    except Exception as e:
        _log_exception(e, "anatomy_guard_apply")

    is_sdxl=("xl" in (current_model_target() or "").lower())

    base_gen=torch.Generator(device=device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    if seed!=0: base_gen.manual_seed(seed)

    cached_embeds=None
    if batch==1:
        disable_first = (os.getenv("PROMPT_CACHE_DISABLE_FIRST_RUN","0")=="1"
                         and run_id==1 and len(_PROMPT_CACHE_ORDER)==0)
        if not disable_first:
            cached_embeds=_get_prompt_embeds(pipe, prompt, negative, cfg, width, height, device)
        else:
            _log_event({"event":"prompt_cache_first_run_skip"})

    images: List[Any]=[]
    step_durations=[]

    with torch.inference_mode():
        for b in range(batch):
            if cancel_event and cancel_event.is_set(): raise RuntimeError("cancelled")
            _set_vae_slicing_for_shape(pipe, width, height, profile)

            g=base_gen
            if seed!=0 and batch>1:
                g=torch.Generator(device=device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
                g.manual_seed(seed+b)
            if progress_cb and b>0: progress_cb(f"Batch {b+1}/{batch}")
            cb=None
            gen_kwargs=dict(num_inference_steps=steps,guidance_scale=cfg,width=width,height=height,
                            generator=g,callback=cb,callback_steps=1)
            try:
                gres = float(anat_meta.get("guidance_rescale", 0.0)) if isinstance(anat_meta, dict) else 0.0
                if gres > 0.0:
                    gen_kwargs["guidance_rescale"] = gres
            except Exception:
                pass
            gen_kwargs["output_type"] = "latent" if profile=="turbo" else "pil"
            if cached_embeds and batch==1:
                pos,neg_emb=cached_embeds
                gen_kwargs.update(prompt_embeds=pos, negative_prompt_embeds=neg_emb)
            else:
                gen_kwargs.update(prompt=prompt, negative_prompt=negative or None)
            t_pipe_call = time.time()
            out = None
            try:
                out = pipe(**gen_kwargs)
            except Exception as e:
                print(f"[ERROR] Pipeline call failed: {e}")
            pipe_ms = int((time.time() - t_pipe_call) * 1000)
            _log_event({"event":"diffusion_call_ms","ms":pipe_ms})

            if profile=="turbo" and out is not None:
                latents = getattr(out, "latents", None) or getattr(out, "images", None)
                if latents is not None:
                    t_dec = time.time()
                    imgs = _sdxl_manual_decode_batch(pipe, latents, force_fp32=False)
                    dec_ms = int((time.time() - t_dec) * 1000)
                    _log_event({"event": "manual_decode_ms", "ms": dec_ms, "count": len(imgs)})
                    images.extend(imgs)
            elif out is not None:
                imgs = getattr(out, 'images', [])
                if (not imgs) and profile == "sdxl":
                    fb = _recover_black_pipeline_image(pipe, prompt, negative, gen_kwargs, profile)
                    if fb is not None:
                        imgs = [fb]
                        _log_event({"event":"sdxl_pipeline_empty_fallback_used"})
                images.extend(imgs)

    sampling_ms = int(sum(step_durations) * 1000)
    total_ms = int((time.time() - t_start) * 1000)
    brightness_list = []
    try:
        import numpy as np
        for im in images:
            brightness_list.append(float(np.asarray(im).mean() / 255.0))
    except Exception:
        pass

    _log_event({"event":"decode_overhead","sampling_ms":sampling_ms,"decode_ms":0,
                 "total_ms":total_ms,"manual":profile=="turbo","run_id":run_id})
    _vram_snapshot("post_generation", {"count":len(images),"w":width,"h":height,"steps":steps})
    _log_event({"event":"generation_summary","count":len(images),"steps":steps,"cfg":cfg,
                "w":width,"h":height,"sampler":sampler,"total_ms":total_ms,
                "brightness":brightness_list,"run_id":run_id})

    if FREE_VRAM_AFTER_GEN:
        release_pipeline(free_ram=False)
    if os.getenv("_ANOMALY_RERUNS_DONE"):
        os.environ["_ANOMALY_RERUNS_DONE"]="0"
    return images

def pil_to_qimage(pil_img):
    """
    Convert a PIL.Image to QImage using PySide6 or PyQt5.
    Raises ImportError if Qt bindings are unavailable.
    """
    try:
        from PySide6 import QtGui  # type: ignore
    except Exception:
        try:
            from PyQt5 import QtGui  # type: ignore
        except Exception as e:
            # Let caller handle; MainWindow suppresses and will show "No images generated."
            raise e
    try:
        from PIL import Image  # noqa: F401
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", pil_img.mode)
        fmt = (QtGui.QImage.Format.Format_RGBA8888
               if pil_img.mode == "RGBA" else QtGui.QImage.Format.Format_RGB888)
        return QtGui.QImage(data, pil_img.width, pil_img.height, fmt).copy()
    except Exception as e:
        # Propagate so the caller's suppress(Exception) keeps qimg_list clean
        raise e

def has_pipeline() -> bool:
    return _PIPELINE is not None

def disk_cache_root() -> str:
    return str(PIPELINE_DISK_CACHE_DIR)
