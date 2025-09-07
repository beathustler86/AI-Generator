from __future__ import annotations

import os
import math
import contextlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops

from src.modules.utils.telemetry import log_event, log_exception

# Global cache for the inpaint pipeline
_PIPE = None
_PIPE_TARGET = None  # model id/path the cache was built for


# ----------------------------
# Configuration and data types
# ----------------------------
@dataclass
class InpaintConfig:
    # Prompts
    prompt: str = ""
    negative_prompt: str = ""

    # Generation controls
    steps: int = 30
    cfg: float = 7.5
    sampler: str = "euler_a"  # accepted by your model loader if you adapt it
    seed: int = 0
    batch: int = 1
    per_batch_seed_increment: int = 0

    # Inpaint behavior
    denoise_strength: float = 0.5  # 0..1
    mode: str = "only_masked"  # "only_masked" | "whole_guided"
    full_res_inpaint: bool = True
    full_res_padding: int = 32  # px

    # Mask post-processing
    feather_radius: float = 0.0  # Gaussian blur radius
    dilate_px: int = 0           # positive to dilate, negative to erode
    invert_mask: bool = False

    # Utilities
    tile_fix: bool = False       # enable VAE tiling to reduce seams/VRAM
    device: str = "cuda"         # "cuda" | "cpu"
    dtype: str = "fp16"          # "fp16" | "fp32"
    precision_mode: str = "FP16" # kept for parity with generation.py

    # Output size override (optional)
    target_width: Optional[int] = None
    target_height: Optional[int] = None


@dataclass
class OverlaySettings:
    show: bool = True
    opacity: float = 0.35  # 0..1


# ----------------------------
# Canvas and mask management
# ----------------------------
class InpaintingCanvas:
    def __init__(self, base_image: Image.Image):
        # Stored in RGB for safety
        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")
        self.image: Image.Image = base_image
        self.mask: Image.Image = Image.new("L", base_image.size, 0)  # 0=keep, 255=inpaint
        self.overlay: OverlaySettings = OverlaySettings()
        self.zoom: float = 1.0
        self.pan: Tuple[int, int] = (0, 0)
        self._brush_size: int = 32
        self._brush_hardness: float = 0.85
        self._brush_opacity: float = 1.0  # 0..1

    # -------- Brush / Eraser ----------
    def set_brush(self, size: int, hardness: float, opacity: float):
        self._brush_size = max(1, int(size))
        self._brush_hardness = float(max(0.0, min(1.0, hardness)))
        self._brush_opacity = float(max(0.0, min(1.0, opacity)))

    def paint(self, x: int, y: int, erase: bool = False):
        """
        Paint or erase onto the mask with a circular brush using hardness and opacity.
        Mask semantics: 255=inpaint area, 0=keep original.
        """
        r = self._brush_size // 2
        if r <= 0:
            return

        # Create a circular stamp with hardness falloff
        stamp = Image.new("L", (self._brush_size, self._brush_size), 0)
        draw = ImageDraw.Draw(stamp)
        draw.ellipse((0, 0, self._brush_size - 1, self._brush_size - 1), fill=255)
        # Hardness shaping: blur the stamp edges based on hardness
        softness = max(0.0, 1.0 - self._brush_hardness)
        if softness > 0:
            # multiply blur radius by brush radius
            blur_radius = max(0.5, softness * r)
            stamp = stamp.filter(ImageFilter.GaussianBlur(blur_radius))

        # Apply opacity
        if self._brush_opacity < 1.0:
            stamp = Image.eval(stamp, lambda px: int(px * self._brush_opacity))

        # Paste onto mask
        mx = int(x - r)
        my = int(y - r)
        region = Image.new("L", self.mask.size, 0)
        region.paste(stamp, (mx, my))

        if erase:
            # Eraser reduces mask coverage
            self.mask = ImageChops.subtract(self.mask, region)
        else:
            # Brush increases mask coverage
            self.mask = ImageChops.lighter(self.mask, region)

    # -------- Selections ----------
    def selection_polygon(self, points: Iterable[Tuple[int, int]], add: bool = True):
        poly_mask = Image.new("L", self.mask.size, 0)
        ImageDraw.Draw(poly_mask).polygon(list(points), fill=255)
        self._merge_selection(poly_mask, add)

    def selection_rectangle(self, x0: int, y0: int, x1: int, y1: int, add: bool = True):
        rect_mask = Image.new("L", self.mask.size, 0)
        ImageDraw.Draw(rect_mask).rectangle((x0, y0, x1, y1), fill=255)
        self._merge_selection(rect_mask, add)

    def _merge_selection(self, sel: Image.Image, add: bool):
        if add:
            self.mask = ImageChops.lighter(self.mask, sel)
        else:
            self.mask = ImageChops.subtract(self.mask, sel)

    # -------- Mask ops ----------
    def apply_feather(self, radius: float):
        radius = max(0.0, float(radius))
        if radius > 0:
            self.mask = self.mask.filter(ImageFilter.GaussianBlur(radius))

    def dilate_erode(self, pixels: int):
        """
        pixels > 0 => dilate by 'pixels'
        pixels < 0 => erode by 'abs(pixels)'
        """
        n = abs(int(pixels))
        if n <= 0:
            return
        filt = ImageFilter.MaxFilter if pixels > 0 else ImageFilter.MinFilter
        # Kernel size must be odd
        k = (2 * n + 1)
        self.mask = self.mask.filter(filt(k))

    def blur_mask(self, radius: float):
        self.apply_feather(radius)

    def invert(self):
        self.mask = ImageChops.invert(self.mask)

    def toggle_overlay(self, show: bool, opacity: Optional[float] = None):
        self.overlay.show = bool(show)
        if opacity is not None:
            self.overlay.opacity = float(max(0.0, min(1.0, opacity)))

    def get_overlay_rgba(self) -> Image.Image:
        """
        Returns an RGBA preview with the mask overlay applied.
        """
        base = self.image.convert("RGBA")
        if not self.overlay.show:
            return base

        alpha = int(max(0, min(255, int(self.overlay.opacity * 255))))
        tint = Image.new("RGBA", base.size, (255, 0, 0, 0))
        # Use mask as alpha to tint
        tint_mask = Image.eval(self.mask, lambda px: int(px * (alpha / 255.0)))
        tint.putalpha(tint_mask)
        return Image.alpha_composite(base, tint)

    # -------- Outpaint ----------
    def outpaint_expand(self, padding: int, fill: str = "edge"):
        """
        Expand canvas by 'padding' pixels on all sides. Mask in expanded area will be 255.
        fill:
          - "edge": extend edge pixels (mirror)
          - "black": pad black
          - "white": pad white
        """
        pad = max(0, int(padding))
        if pad == 0:
            return
        w, h = self.image.size
        new_w, new_h = w + 2 * pad, h + 2 * pad

        if fill == "edge":
            # Mirror by tiling and cropping trick
            mirrored = ImageOps.expand(self.image, border=pad, fill=None)
            mirrored = ImageOps.mirror(mirrored.crop((0, 0, new_w, new_h)))
            # That basic approach can look odd; as a simple alternative, use edge pixels
            # A simpler approximation: paste original on solid background, then blur edges
            bg = Image.new("RGB", (new_w, new_h), self.image.getpixel((0, 0)))
            bg.paste(self.image, (pad, pad))
            bg = bg.filter(ImageFilter.BoxBlur(2))
            self.image = bg
        else:
            color = (0, 0, 0) if fill == "black" else (255, 255, 255)
            bg = Image.new("RGB", (new_w, new_h), color)
            bg.paste(self.image, (pad, pad))
            self.image = bg

        # Expand mask, new area = 255 (inpaint)
        new_mask = Image.new("L", (new_w, new_h), 255)
        new_mask.paste(self.mask, (pad, pad))
        self.mask = new_mask


# ----------------------------
# Pipeline management
# ----------------------------
def ensure_pipeline(model_path: Optional[str] = None,
                    device: str = "cuda",
                    dtype: str = "fp16"):
    """
    Lazily load and cache StableDiffusionXLInpaintPipeline (Diffusers).
    """
    global _PIPE, _PIPE_TARGET

    # Determine target path
    target = model_path or os.environ.get("INPAINT_MODEL_PATH") \
             or os.environ.get("MODEL_INPAINT_PATH") \
             or os.environ.get("MODEL_ID_OR_PATH")  # fallback

    if _PIPE is not None and _PIPE_TARGET == target:
        return _PIPE

    try:
        from diffusers import StableDiffusionXLInpaintPipeline
        import torch

        log_event({"event": "inpaint_pipe_load_start", "target": target})
        torch_dtype = torch.float16 if dtype.lower() in ("fp16", "float16") else torch.float32
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            target,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        pipe = pipe.to(device)

        # Defaults similar to SDXL turbo handling: attention slicing off by default
        if hasattr(pipe, "enable_vae_slicing"):
            # Leave off unless memory constrained
            pass
        if hasattr(pipe, "enable_vae_tiling"):
            # Enable on demand via 'tile_fix'
            pass

        _PIPE = pipe
        _PIPE_TARGET = target
        log_event({"event": "inpaint_pipe_load_ok", "device": device, "dtype": str(torch_dtype)})
        return _PIPE
    except Exception as e:
        log_exception(e, context="inpaint_pipe_load")
        raise


def release_pipeline():
    global _PIPE, _PIPE_TARGET
    try:
        if _PIPE is not None:
            with contextlib.suppress(Exception):
                _PIPE.to("cpu")
            _PIPE = None
            _PIPE_TARGET = None
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
        log_event({"event": "inpaint_pipe_released"})
    except Exception as e:
        log_exception(e, context="inpaint_pipe_release")


# ----------------------------
# Inpaint entrypoint
# ----------------------------
def inpaint_image(
    canvas: InpaintingCanvas,
    cfg: InpaintConfig,
    progress_cb: Callable[[str], None] = lambda m: None,
    cancel_event=None
) -> List[Image.Image]:
    """
    Perform SDXL inpainting given a canvas and configuration.
    Returns list of PIL Images (RGB).
    """
    # Early cancel
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return []

    # Prepare base and mask
    image = canvas.image
    mask = canvas.mask

    if cfg.invert_mask:
        mask = ImageOps.invert(mask)

    # Mask post-processing
    if cfg.dilate_px != 0:
        _tmp = InpaintingCanvas(image)
        _tmp.mask = mask
        _tmp.dilate_erode(cfg.dilate_px)
        mask = _tmp.mask

    if cfg.feather_radius > 0.0:
        mask = mask.filter(ImageFilter.GaussianBlur(cfg.feather_radius))

    # Optional full-res inpaint: upscale masked crop to full image for better detail
    # (Scaffold: keep full image; crop/pad logic can be added later)
    width, height = image.size
    tgt_w = int(cfg.target_width or width)
    tgt_h = int(cfg.target_height or height)
    if (tgt_w, tgt_h) != (width, height):
        image = image.resize((tgt_w, tgt_h), Image.LANCZOS)
        mask = mask.resize((tgt_w, tgt_h), Image.NEAREST)

    # Load pipeline (lazy)
    progress_cb("Loading inpaint pipeline...")
    with contextlib.suppress(Exception):
        log_event({"event": "inpaint_begin", "w": image.width, "h": image.height, "steps": cfg.steps, "cfg": cfg.cfg})
    pipe = ensure_pipeline(device=cfg.device, dtype=cfg.dtype)

    # Tile fix: enable VAE tiling during decode to reduce seams/VRAM
    if cfg.tile_fix and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    else:
        with contextlib.suppress(Exception):
            pipe.disable_vae_tiling()  # may not exist on older diffusers

    # Sampler: keep consistent with your generation module if needed
    # For now we rely on pipeline defaults. You can adapt scheduler based on cfg.sampler here.

    images: List[Image.Image] = []
    cur_seed = int(cfg.seed or 0)
    for i in range(max(1, int(cfg.batch))):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            progress_cb("Cancelled")
            break

        # Seed handling
        generator = None
        try:
            import torch
            if cur_seed > 0:
                generator = torch.Generator(device=cfg.device).manual_seed(int(cur_seed))
        except Exception:
            pass

        progress_cb(f"Sampling batch {i+1}/{cfg.batch}...")
        try:
            # Stable Diffusion XL Inpaint expects:
            #   image: PIL.Image (RGB), mask_image: PIL.Image (L/1), denoising_strength: 0..1
            # Guidance scale = cfg, num_inference_steps = steps
            res = pipe(
                prompt=cfg.prompt or "",
                negative_prompt=cfg.negative_prompt or None,
                image=image,
                mask_image=mask,
                guidance_scale=float(cfg.cfg),
                num_inference_steps=int(cfg.steps),
                denoising_strength=float(max(0.0, min(1.0, cfg.denoise_strength))),
                generator=generator
            )
            pil = res.images[0].convert("RGB")
        except Exception as e:
            log_exception(e, context="inpaint_run")
            raise

        # Post-compose with original depending on mode
        if cfg.mode == "only_masked":
            # Keep original outside mask
            pil = Image.composite(pil, image, mask)
        else:
            # whole_guided: lightly blend original outside to reduce global drift
            soft = mask.filter(ImageFilter.GaussianBlur(2.0))
            inv = ImageOps.invert(soft)
            pil = Image.composite(pil, image, inv)

        images.append(pil)

        if cfg.per_batch_seed_increment:
            cur_seed = (cur_seed + int(cfg.per_batch_seed_increment)) if cur_seed else 0

    progress_cb("Finalizing")
    with contextlib.suppress(Exception):
        log_event({"event": "inpaint_done", "count": len(images)})

    return images


# ----------------------------
# Utilities for EXIF strip and zoom/pan placeholders
# ----------------------------
def strip_exif(img: Image.Image) -> Image.Image:
    try:
        data = list(img.getdata())
        no_exif = Image.new(img.mode, img.size)
        no_exif.putdata(data)
        return no_exif
    except Exception:
        # Fallback: return a copy
        return img.copy()


# Optional: helpers to assist a future GUI window
def safe_open_image(path: str, strip: bool = True) -> Image.Image:
    im = Image.open(path)
    im = im.convert("RGB")
    if strip:
        im = strip_exif(im)
    return im
