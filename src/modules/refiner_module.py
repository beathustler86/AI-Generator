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
  SDXL_REFINER_GUIDANCE    RGB fallback guidance scale (default 3.5)
  SDXL_REFINER_ASYNC=1     Allow background async loading (default 1)
"""

from __future__ import annotations
import os, json, time, threading, traceback
from datetime import datetime
from typing import Optional, Any, Dict, Tuple
import torch
from PIL import Image
import contextlib

try:
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
    _HAS_SDXL_REFINER = True
except Exception:
    StableDiffusionXLRefinerPipeline = None  # type: ignore
    _HAS_SDXL_REFINER = False

# Auto pipeline fallback (covers builds where refiner class isn't top-level-exported)
try:
    from diffusers import AutoPipelineForImage2Image  # type: ignore
    _HAS_AUTO_I2I = True
except Exception:
    AutoPipelineForImage2Image = None  # type: ignore
    _HAS_AUTO_I2I = False

# Img2Img fallbacks (no refiner)
try:
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline  # type: ignore
except Exception:
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore
    StableDiffusionImg2ImgPipeline = None  # type: ignore

# ---------- Paths / Config ----------
_REPO_DEFAULT = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/stable-diffusion-xl-refiner-1.0"
REFINER_PATH = os.environ.get("SDXL_REFINER_PATH", _REPO_DEFAULT)
_SPLIT_DEF = float(os.environ.get("SDXL_REFINER_SPLIT", "0.8"))
_STEPS_DEF = int(os.environ.get("SDXL_REFINER_STEPS", "6"))
_STRENGTH_DEF = float(os.environ.get("SDXL_REFINER_STRENGTH", "0.35"))
_GUIDANCE_DEF = float(os.environ.get("SDXL_REFINER_GUIDANCE", "3.5"))
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

_IMPORTABLE = bool(_HAS_SDXL_REFINER or _HAS_AUTO_I2I)

# ---------- Core Helpers ----------
def available() -> bool:
    # Path exists AND we have either the native refiner class or AutoPipeline fallback
    return os.path.isdir(REFINER_PATH) and (_HAS_SDXL_REFINER or _HAS_AUTO_I2I)

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

def _enable_tf32():
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Prefer faster matmul kernels on Ampere+
            with contextlib.suppress(Exception):
                torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _get_active_pipeline():
    # Try to reuse the currently loaded generation pipeline (for Img2Img fallback)
    try:
        from src.modules import generation as gen_mod  # type: ignore
        return getattr(gen_mod, "_PIPELINE", None)
    except Exception:
        return None

def _pil_cleanup(image: Image.Image) -> Image.Image:
    # Gentle cleanup: autocontrast -> median denoise -> unsharp mask
    try:
        from PIL import ImageFilter, ImageOps
        img = image.convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=1)  # clip 1% for mild contrast fix
        img = img.filter(ImageFilter.MedianFilter(size=3))  # mild denoise
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))  # subtle sharpening
        return img
    except Exception as e:
        _log_exception(e, "pil_cleanup")
        return image

def _fallback_img2img(
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str],
    steps: int,
    strength: float,
    guidance_scale: float
) -> Optional[Image.Image]:
    """
    Build an Img2Img pipeline from the active T2I pipeline components and run a low-strength cleanup pass.
    """
    pipe = _get_active_pipeline()
    if pipe is None:
        _log("refiner_img2img_no_active_pipe")
        return None

    # Resolve device/dtype from UNet if possible
    try:
        ref_param = next(pipe.unet.parameters())
        dev = "cuda" if ref_param.is_cuda else "cpu"
        dt = ref_param.dtype
    except Exception:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32

    # Pick proper Img2Img class
    try:
        from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline  # type: ignore
        is_xl = isinstance(pipe, StableDiffusionXLPipeline)
    except Exception:
        is_xl = hasattr(pipe, "text_encoder_2")  # heuristic

    img2img_cls = StableDiffusionXLImg2ImgPipeline if is_xl else StableDiffusionImg2ImgPipeline
    if img2img_cls is None:
        _log("refiner_img2img_class_missing", is_xl=is_xl)
        return None

    t0 = time.time()
    try:
        # Construct from components to avoid reloading weights
        img2img = img2img_cls(**pipe.components)  # type: ignore[arg-type]
        img2img = img2img.to(dev)
        # Keep dtype alignment to avoid casts
        with torch.inference_mode():
            try:
                # Some pipelines support full .to(dtype=)
                img2img.to(dtype=dt)
            except Exception:
                pass
        _enable_xformers(img2img)

        # Run cleanup pass
        with torch.inference_mode():
            out = img2img(
                prompt=prompt or "",
                negative_prompt=negative_prompt,
                image=image.convert("RGB"),
                strength=float(max(0.05, min(0.95, strength))),
                num_inference_steps=int(max(1, steps)),
                guidance_scale=float(max(0.0, guidance_scale))
            )
        res = out.images[0] if hasattr(out, "images") and out.images else None
        if res is not None:
            _log("refiner_rgb_fallback_img2img_ok",
                 is_xl=is_xl, steps=steps, strength=strength, guidance=guidance_scale,
                 ms=int((time.time()-t0)*1000))
        else:
            _log("refiner_rgb_fallback_img2img_empty")
        return res
    except Exception as e:
        _log_exception(e, "refiner_img2img_run")
        return None

# ---------- Load ----------
def load_refiner(force: bool = False):
    global _pipe, _device, _dtype
    if not os.path.isdir(REFINER_PATH):
        _log("refiner_unavailable", path=REFINER_PATH, importable=False)
        return None
    with _lock:
        if _pipe is not None and not force:
            return _pipe
        dev, dt = _choose_device()
        variant = os.environ.get("SDXL_REFINER_VARIANT", "fp16")
        t0 = time.time()
        try:
            if _HAS_SDXL_REFINER and StableDiffusionXLRefinerPipeline is not None:
                pipe = StableDiffusionXLRefinerPipeline.from_pretrained(
                    REFINER_PATH,
                    torch_dtype=dt,
                    use_safetensors=True,
                    safety_checker=None,
                    variant=variant,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
            elif _HAS_AUTO_I2I and AutoPipelineForImage2Image is not None:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    REFINER_PATH,
                    torch_dtype=dt,
                    use_safetensors=True,
                    variant=variant,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
            else:
                _log("refiner_no_loader_available")
                return None

            if dev == "cuda":
                _enable_tf32()
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
    """
    Preferred: use SDXL Refiner if available.
    Fallback: Img2Img with the active pipeline (low-strength cleanup).
    Final fallback: gentle PIL cleanup (autocontrast/denoise/sharpen).
    """
    pipe = load_refiner()
    if pipe is None:
        _log("refiner_rgb_no_pipe")
        # Try Img2Img fallback using active pipeline
        img2img_steps = int(os.environ.get("SDXL_REFINER_STEPS", str(max(1, steps))))
        img2img_strength = float(os.environ.get("SDXL_REFINER_STRENGTH", str(max(0.05, min(0.95, strength)))))
        img2img_guidance = float(os.environ.get("SDXL_REFINER_GUIDANCE", str(_GUIDANCE_DEF)))
        res = _fallback_img2img(
            image=image,
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            steps=img2img_steps,
            strength=img2img_strength,
            guidance_scale=img2img_guidance
        )
        if res is not None:
            return res
        # Last-resort PIL cleanup
        cleaned = _pil_cleanup(image)
        _log("refiner_rgb_fallback_pil_ok")
        return cleaned

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
        # Attempt Img2Img fallback even if refiner load failed at run time
        res = _fallback_img2img(
            image=image,
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            steps=int(max(1, steps)),
            strength=float(max(0.05, min(0.95, strength))),
            guidance_scale=_GUIDANCE_DEF
        )
        if res is not None:
            return res
        return _pil_cleanup(image)

# ---------- Back-compat UI Wrapper ----------
def refine_image(image, prompt=None, negative=None, **_kw_unused):
    """
    Returns dict: {"image": PIL.Image or None}
    Uses recommended defaults for cleanup when refiner is unavailable.
    """
    try:
        out_img = refine_image_from_rgb(
            image,
            prompt=prompt or "",
            negative_prompt=negative,
            steps=int(os.getenv("SDXL_REFINER_STEPS", "10")),
            strength=float(os.getenv("SDXL_REFINER_STRENGTH", "0.35"))
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
        "importable": _IMPORTABLE,  # now defined
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
