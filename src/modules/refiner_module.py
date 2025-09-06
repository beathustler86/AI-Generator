"""
SDXL Refiner module (latent handoff + optional RGB fallback)

Usage (latent path – higher fidelity):
  base_out = base_pipe(
      prompt=..., negative_prompt=..., num_inference_steps=BASE_STEPS,
      guidance_scale=CFG, width=W, height=H,
      output_type="latent", denoising_end=SPLIT)
  latents = base_out.images[0]  # (B,C,H,W)
  refined = refine_from_latents(latents, prompt, negative_prompt, guidance_scale=CFG)

Fallback (RGB) if latents unavailable:
  refined_img = refine_image_from_rgb(pil_image, prompt, negative_prompt)

Environment variables:
  SDXL_REFINER_PATH        Path to refiner weights (default: <repo>/models/.../sdxl-refiner-1.0)
  SDXL_REFINER_SPLIT       Default split (denoising start) for latent refine (default 0.8)
  SDXL_REFINER_STEPS       Refiner steps (default 6)
  SDXL_REFINER_VARIANT     'fp16' / 'fp32' / etc. (default fp16)
  SDXL_REFINER_XFORMERS=1  Enable memory‑efficient attention if available (default 1)
  SDXL_REFINER_STRENGTH    RGB fallback strength (default 0.35)
  SDXL_REFINER_ASYNC=1     Allow background async loading (default 1)
"""

from __future__ import annotations
import os, json, time, threading, traceback
from datetime import datetime
from typing import Optional, Any, Dict, Tuple
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
    _IMPORTABLE = True
except Exception:
    StableDiffusionXLRefinerPipeline = None  # type: ignore
    _IMPORTABLE = False

# ---------- Paths / Config ----------
_REPO_DEFAULT = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/stable-diffusion-xl-refiner-1.0"
REFINER_PATH = os.environ.get("SDXL_REFINER_PATH", _REPO_DEFAULT)
_SPLIT_DEF = float(os.environ.get("SDXL_REFINER_SPLIT", "0.8"))
_STEPS_DEF = int(os.environ.get("SDXL_REFINER_STEPS", "6"))
_STRENGTH_DEF = float(os.environ.get("SDXL_REFINER_STRENGTH", "0.35"))
_ASYNC = os.environ.get("SDXL_REFINER_ASYNC", "1") == "1"

# ---------- Telemetry helpers ----------
def _now(): return datetime.utcnow().isoformat()

def _fallback_telemetry(rec: Dict[str, Any]):
    # Local file fallback if telemetry module absent
    try:
        log_root = os.path.join("logs")
        os.makedirs(log_root, exist_ok=True)
        with open(os.path.join(log_root, "refiner_fallback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

try:
    from src.modules.utils.telemetry import log_event as _log_event, log_exception as _log_exception
except Exception:
    def _log_event(d: Dict[str, Any]): _fallback_telemetry(d)
    def _log_exception(e: Exception, context: str = ""):
        _fallback_telemetry({"event":"exception","ctx":context,"err":str(e),"trace":traceback.format_exc(),"ts":_now()})

def _log(event: str, **kw):
    kw.update(event=event, ts=_now())
    _log_event(kw)

# ---------- Internal State ----------
_lock = threading.RLock()
_pipe: Optional[Any] = None
_device: Optional[str] = None
_dtype: Optional[torch.dtype] = None
_async_thread: Optional[threading.Thread] = None

# ---------- Core Helpers ----------
def available() -> bool:
    return _IMPORTABLE and os.path.isdir(REFINER_PATH)

def has_refiner() -> bool:
    return _pipe is not None or available()

def _choose_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32

def _enable_xformers(pipe):
    if os.environ.get("SDXL_REFINER_XFORMERS","1") != "1": return
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            _log("refiner_xformers_enabled")
    except Exception as e:
        _log_exception(e, "refiner_xformers")

# ---------- Load ----------
def load_refiner(force: bool = False):
    global _pipe, _device, _dtype
    if not available():
        _log("refiner_unavailable", path=REFINER_PATH, importable=_IMPORTABLE)
        return None
    with _lock:
        if _pipe is not None and not force:
            return _pipe
        dev, dt = _choose_device()
        variant = os.environ.get("SDXL_REFINER_VARIANT", "fp16")
        t0 = time.time()
        try:
            pipe = StableDiffusionXLRefinerPipeline.from_pretrained(
                REFINER_PATH,
                torch_dtype=dt,
                use_safetensors=True,
                safety_checker=None,
                variant=variant
            )
            if dev == "cuda":
                pipe.to(dev)
            _enable_xformers(pipe)
            _pipe = pipe
            _device = dev
            _dtype = dt
            _log("refiner_load", device=dev, variant=variant,
                 ms=int((time.time()-t0)*1000))
            return _pipe
        except Exception as e:
            _log_exception(e, "refiner_load")
            _pipe = None
            return None

def _async_loader():
    try:
        load_refiner()
    except Exception as e:
        _log_exception(e, "refiner_async")

def ensure_loaded_async():
    global _async_thread
    if not _ASYNC or _pipe is not None:
        return
    if _async_thread and _async_thread.is_alive():
        return
    _async_thread = threading.Thread(target=_async_loader, daemon=True)
    _async_thread.start()
    _log("refiner_async_start")

# ---------- Refinement (Latent) ----------
def refine_from_latents(
    latents: torch.Tensor,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float = 5.0,
    split: Optional[float] = None,
    steps: Optional[int] = None
):
    pipe = load_refiner()
    if pipe is None:
        _log("refiner_latent_no_pipe")
        return None
    if latents is None or not torch.is_tensor(latents):
        _log("refiner_latent_invalid")
        return None
    if latents.dim() != 4:
        _log("refiner_latent_badshape", shape=list(latents.shape))
        return None
    use_split = split if split is not None else _SPLIT_DEF
    use_split = min(max(use_split, 0.5), 0.95)
    use_steps = steps if steps is not None else _STEPS_DEF
    use_steps = max(1, use_steps)
    t0 = time.time()
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=use_steps,
            denoising_start=use_split,
            guidance_scale=guidance_scale,
            latents=latents
        )
        img = out.images[0]
        _log("refiner_latent_ok", split=use_split, steps=use_steps,
             ms=int((time.time()-t0)*1000), guidance=guidance_scale)
        return img
    except Exception as e:
        _log_exception(e, "refiner_latent")
        return None

# ---------- RGB Fallback ----------
def refine_image_from_rgb(
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 10,
    strength: float = 0.3
):
    pipe = load_refiner()
    if pipe is None:
        _log("refiner_rgb_no_pipe")
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")
    t0 = time.time()
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps
        )
        res = out.images[0]
        _log("refiner_rgb_ok", steps=steps, strength=strength,
             ms=int((time.time()-t0)*1000))
        return res
    except Exception as e:
        _log_exception(e, "refiner_rgb")
        return image

# ---------- Back-compat UI Wrapper ----------
def refine_image(image, prompt=None, negative=None, **kw):
    """
    Returns dict: {"image": PIL.Image or None}
    """
    try:
        out_img = refine_image_from_rgb(
            image,
            prompt=prompt or "",
            negative_prompt=negative,
            steps=int(os.getenv("SDXL_REFINER_STEPS", str(_STEPS_DEF))),
            strength=float(os.getenv("SDXL_REFINER_STRENGTH", str(_STRENGTH_DEF)))
        )
        return {"image": out_img}
    except Exception as e:
        _log_exception(e, "refine_image_wrapper")
        return {"image": None, "error": str(e)}

# ---------- Status ----------
def status() -> Dict[str, Any]:
    return {
        "available": available(),
        "loaded": _pipe is not None,
        "device": _device,
        "dtype": str(_dtype),
        "path": REFINER_PATH,
        "importable": _IMPORTABLE,
        "ts": _now()
    }

def get_refiner_status() -> Dict[str, Any]:
    return status()

__all__ = [
    "available",
    "has_refiner",
    "ensure_loaded_async",
    "load_refiner",
    "refine_from_latents",
    "refine_image_from_rgb",
    "refine_image",
    "status",
    "get_refiner_status"
]
