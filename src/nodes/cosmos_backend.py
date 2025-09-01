"""
Cosmos / ComfyUI video backend probe & wrapper.

Decision order:
 1. If explicit env COSMOS_FORCE_STUB=1 -> stub.
 2. Try importing required classes (GeneralDIT, FinalLayer).
 3. Check for weights directory (env COSMOS_WEIGHTS or models/text_to_video/Cosmos).
 4. (Placeholder) load minimal model (not implemented yet) -> if fails, fallback.

Telemetry events emitted (use VERBOSE_TELEMETRY=1 to see all):
  - video_backend_probe
  - video_backend_unavailable_reason
  - video_backend_selected
"""
from __future__ import annotations
from pathlib import Path
import os, math, random, textwrap
from typing import Optional, List, Dict, Any

from PIL import Image, ImageDraw, ImageFont
import torch

try:
    from src.modules.utils.telemetry import log_event, log_exception
except Exception:
    def log_event(d): pass
    def log_exception(e, context=""): pass

_BACKEND_CACHE: Dict[str, Any] = {}

def _emit(event: str, **kw):
    log_event({"event": event, **kw})

def probe_backend() -> str:
    if os.environ.get("COSMOS_FORCE_STUB") == "1":
        _emit("video_backend_selected", backend="stub", forced=True)
        return "stub"
    if "backend" in _BACKEND_CACHE:
        return _BACKEND_CACHE["backend"]
    _emit("video_backend_probe")
    try:
        from src.nodes.cosmos.model import GeneralDIT  # type: ignore
    except Exception as e:
        _emit("video_backend_unavailable_reason", stage="import", detail=str(e))
        _BACKEND_CACHE["backend"] = "stub"
        return "stub"
    weights_root = os.environ.get("COSMOS_WEIGHTS")
    if not weights_root:
        candidate = Path("models") / "text_to_video" / "Cosmos"
        if candidate.exists():
            weights_root = str(candidate.resolve())
    if not weights_root:
        _emit("video_backend_unavailable_reason", stage="weights_path", detail="no COSMOS_WEIGHTS or default path")
        _BACKEND_CACHE["backend"] = "real_pending"
        return "real_pending"
    wr = Path(weights_root)
    weight_files = list(wr.glob("*.safetensors")) + list(wr.glob("*.pt")) + list(wr.glob("*.bin"))
    if not weight_files:
        _emit("video_backend_unavailable_reason", stage="weights_missing", detail=weights_root)
        _BACKEND_CACHE["backend"] = "real_pending"
        _BACKEND_CACHE["weights_root"] = weights_root
        return "real_pending"
    # Attempt lightweight model build once
    if _load_real_model():
        _BACKEND_CACHE["backend"] = "real_ready"
        _emit("video_backend_selected", backend="real_ready", weights=weights_root)
    else:
        _BACKEND_CACHE["backend"] = "real_pending"
    _BACKEND_CACHE["weights_root"] = weights_root
    return _BACKEND_CACHE["backend"]

def _load_real_model() -> bool:
    if "model" in _BACKEND_CACHE:
        return True
    try:
        from src.nodes.cosmos.model import GeneralDIT  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Minimal tiny config (placeholder) – replace with real values / load weights.
        model = GeneralDIT(
            max_img_h=64, max_img_w=64, max_frames=8,
            in_channels=3, out_channels=3,
            patch_spatial=4, patch_temporal=2,
            block_config="FA-CA-MLP",
            model_channels=128, num_blocks=2, num_heads=4,
            crossattn_emb_channels=256
        ).to(device)
        model.eval()
        _BACKEND_CACHE["model"] = model
        _BACKEND_CACHE["device"] = device
        return True
    except Exception as e:
        log_exception(e, context="cosmos_model_build")
        _emit("video_backend_unavailable_reason", stage="model_build", detail=str(e))
        return False

def backend_status() -> Dict[str, Any]:
    probe = probe_backend()
    return {
        "backend": probe,
        "weights_root": _BACKEND_CACHE.get("weights_root"),
    }

# ---------- Frame generation entry point ----------

def _diffusion_stub(model, frames, width, height, seed, progress_cb, cancel_event=None):
    """
    Simple diffusion-like loop (placeholder). Creates latent noise and 'denoises'
    by smoothing over steps. Replace with real scheduler + UNet forward.
    """
    import torch
    steps = min(30, max(5, frames//3))
    rng = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, 3, frames, height//8, width//8, generator=rng)
    for s in range(steps):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")
        latent = latent * 0.9  # pretend denoise
        if progress_cb and (s % max(1, steps//10) == 0 or s == steps-1):
            progress_cb(f"Step {s+1}/{steps}")
    # Upsample (nearest) back to target
    latent = torch.nn.functional.interpolate(
        latent.view(1,3,frames,height//8,width//8).permute(0,2,1,3,4).reshape(frames,3,height//8,width//8),
        scale_factor=8, mode="nearest")
    frames_out=[]
    for i in range(frames):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")
        arr = latent[i].clamp(-3,3)
        arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
        img = (arr.permute(1,2,0).numpy()*255).astype("uint8")
        frames_out.append(Image.fromarray(img))
    return frames_out

def generate_frames(prompt: str, frames: int, width: int, height: int,
                    seed: int, fps: int, progress_cb=None, cancel_event=None,
                    steps: Optional[int] = None, cfg: Optional[float] = None,
                    negative: str = ""):
    backend = probe_backend()
    sampling_flag = os.environ.get("COSMOS_SAMPLING","0") == "1"
    try:
        if backend == "real_ready":
            # Prefer real pipeline if loaded with sampling flag, else attempt actual cosmos pipeline.
            if sampling_flag:
                return _diffusion_stub(_BACKEND_CACHE["model"], frames, width, height, seed, progress_cb, cancel_event)
            try:
                from .cosmos_pipeline import CosmosPipeline
                pipe = _BACKEND_CACHE.get("cosmos_pipe")
                if pipe is None:
                    pipe = CosmosPipeline.from_environment(_BACKEND_CACHE.get("weights_root"))
                    _BACKEND_CACHE["cosmos_pipe"] = pipe
                return pipe.generate(prompt=prompt,
                                     negative_prompt=negative,
                                     num_frames=frames,
                                     width=width,
                                     height=height,
                                     steps=steps or 30,
                                     guidance_scale=cfg or 7.5,
                                     seed=seed,
                                     fps=fps,
                                     progress_cb=progress_cb,
                                     cancel_event=cancel_event)
            except Exception as e:
                log_exception(e, context="cosmos_pipeline")
                _emit("video_backend_unavailable_reason", stage="cosmos_pipeline", detail=str(e))
                # Fallback to stub noise style (real_pending look) so user still sees something.
                return _synthetic(prompt, frames, width, height, seed, fps, progress_cb, cancel_event, style="noise+text")
        if backend == "real_pending":
            _emit("video_backend_state", state="real_pending")
            return _synthetic(prompt, frames, width, height, seed, fps, progress_cb, cancel_event, style="noise+text")
        if backend == "stub":
            _emit("video_backend_state", state="stub")
            return _synthetic(prompt, frames, width, height, seed, fps, progress_cb, cancel_event, style="ball+text")
        _emit("video_backend_warning", state=backend)
        return _synthetic(prompt, frames, width, height, seed, fps, progress_cb, cancel_event, style="ball+text")
    except RuntimeError as e:
        if str(e) == "cancelled":
            raise
        raise

def _model_infer(prompt: str, frames: int, width: int, height: int,
                 seed: int, progress_cb):
    model = _BACKEND_CACHE["model"]
    device = _BACKEND_CACHE["device"]
    torch.manual_seed(seed)
    # Resize (pad/crop) dims to multiples of patch size used in placeholder (4)
    def _round(x, m): return (x // m) * m
    H = max(32, _round(height, 4))
    W = max(32, _round(width, 4))
    T = min(frames, model.max_frames) if hasattr(model, "max_frames") else min(frames, 8)
    x = torch.randn(1, 3, T, H, W, device=device)
    timesteps = torch.randint(low=0, high=999, size=(1,), device=device)
    context = torch.zeros(1, 4, model.model_channels if hasattr(model,"model_channels") else 128, device=device)
    with torch.no_grad():
        out = model(x, timesteps=timesteps, context=context)
    # out shape: (B,C,T,H,W) expected
    if out.ndim != 5:
        out = x  # fallback
    vid = out[0].detach().float().cpu()
    vid = (vid - vid.min()) / (vid.max() - vid.min() + 1e-8)
    import numpy as np
    pil_frames: List[Image.Image] = []
    for i in range(T):
        frame = vid[:, i, :, :] if vid.shape[1] == 3 else vid[i]
        arr = (frame.numpy() * 255).clip(0,255).astype("uint8")
        if arr.shape[0] == 3:
            arr = arr.transpose(1,2,0)
        else:
            arr = arr
        img = Image.fromarray(arr)
        pil_frames.append(img)
        if progress_cb and (i % max(1, T//10) == 0 or i == T-1):
            progress_cb(f"Frame {i+1}/{T}")
    if frames > T:
        # Loop-fill if user requested more
        extra = []
        for i in range(frames - T):
            extra.append(pil_frames[i % T].copy())
            if progress_cb and ((T+i) % max(1, frames//10) == 0 or (T+i)==frames-1):
                progress_cb(f"Frame {T+i+1}/{frames}")
        pil_frames.extend(extra)
    return pil_frames

# ---------- Synthetic generators ----------

def _synthetic(prompt: str, frames: int, width: int, height: int, seed: int,
               fps: int, progress_cb, cancel_event, style="ball+text"):
    rnd = random.Random(seed)
    base_color = tuple(rnd.randint(50, 200) for _ in range(3))
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = None
    lines = textwrap.wrap(prompt or "video", width=60)[:4]
    imgs: List[Image.Image] = []
    radius = max(6, min(width, height)//18)
    for i in range(frames):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")
        img = Image.new("RGB", (width, height), (0,0,0))
        draw = ImageDraw.Draw(img)
        t = i / max(1, frames-1)
        if style.startswith("ball"):
            x = int((width - 2*radius) * abs(math.sin(math.pi * t)))
            y = int((height - 2*radius) * abs(math.sin(math.pi * 1.3 * t)))
            draw.ellipse([x, y, x+2*radius, y+2*radius], fill=base_color)
        else:
            block = 32
            for bx in range(0, width, block):
                for by in range(0, height, block):
                    shade = int((rnd.random()*0.6 + 0.2)*255*(0.3+0.7*math.sin( (i+1+bx+by)/13)))
                    draw.rectangle([bx, by, bx+block-1, by+block-1], fill=(shade,shade,shade))
        ty = height - (len(lines)*16 + 4)
        for li, line in enumerate(lines):
            draw.text((6, ty + li*16), line, fill=(210,210,210), font=font)
        imgs.append(img)
        if progress_cb and (i % max(1, frames//12) == 0 or i == frames-1):
            progress_cb(f"Frame {i+1}/{frames}")
    return imgs
