"""
Refiner subsystem — robust, single-definition version.
No future imports (removed to avoid SyntaxError triggered by hidden BOM / cache).
"""
import os, sys, json, threading, traceback, time
from datetime import datetime
from typing import Optional, Any, Dict

from PIL import Image
import torch

REFINER_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-refiner-1.0"
SAVE_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/output/refined"
PROMPT = "cockpit-grade GUI with tactical overlays"

_TELEMETRY_LOG = os.path.join(
    "F:/SoftwareDevelopment/AI Models Image/AIGenerator/outputs",
    "logs", "telemetry_logs", "telemetry.jsonl"
)

try:
    from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore
    REFINER_IMPORTABLE = True
except Exception:
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore
    REFINER_IMPORTABLE = False

REFINER_AVAILABLE = os.path.isdir(REFINER_PATH) and REFINER_IMPORTABLE

_refiner_lock = threading.RLock()
_refiner_pipe: Optional[Any] = None
_refiner_device: Optional[str] = None

def _safe_write_jsonl(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

def log_event(event: Any) -> None:
    if isinstance(event, dict):
        payload = event.copy()
        payload.setdefault("timestamp", datetime.now().isoformat())
    else:
        payload = {"message": str(event), "timestamp": datetime.now().isoformat()}
    _safe_write_jsonl(_TELEMETRY_LOG, payload)

def log_telemetry(**kw):
    kw.setdefault("timestamp", datetime.now().isoformat())
    log_event(kw)

def log_memory(device: str):
    try:
        if device == "cuda" and torch.cuda.is_available():
            mb = torch.cuda.memory_allocated() / 1024**2
            log_event({"event": "RefinerMemory", "device": device, "allocated_mb": round(mb, 2)})
    except Exception:
        pass

def _serialize_config(pipe) -> Dict[str, Any]:
    try:
        cfg = getattr(pipe, "config", None)
        if cfg is None:
            return {"config": None}
        try:
            return dict(cfg)
        except Exception:
            import json as _json
            try:
                return _json.loads(_json.dumps(cfg, default=str))
            except Exception:
                return {"config_str": str(cfg)}
    except Exception:
        return {"config_error": "failed to serialize config"}

def load_refiner(device: str = "cuda", force_reload: bool = False):
    global _refiner_pipe, _refiner_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if not REFINER_AVAILABLE:
        if _refiner_pipe is None:
            _refiner_pipe = object()
            _refiner_device = "stub"
        return _refiner_pipe
    with _refiner_lock:
        if _refiner_pipe is not None and not force_reload and _refiner_device == device:
            return _refiner_pipe
        try:
            print("[Refiner] Loading img2img model...", flush=True)
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                REFINER_PATH,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16",
                use_safetensors=True
            ).to(device)
            _refiner_pipe = pipe
            _refiner_device = device
            log_event({"event": "RefinerLoad", "status": "success",
                       "device": device, "path": REFINER_PATH,
                       "config": _serialize_config(pipe)})
            print("[Refiner] Ready.", flush=True)
            return _refiner_pipe
        except Exception as e:
            log_event({"event": "RefinerLoadFailed", "device": device, "error": str(e),
                       "trace": traceback.format_exc()})
            print(f"[Refiner] Load failed on {device}: {e}", flush=True)
            if device != "cpu":
                try:
                    print("[Refiner] Retrying on CPU...", flush=True)
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        REFINER_PATH,
                        torch_dtype=torch.float32,
                        variant="fp16",
                        use_safetensors=True
                    ).to("cpu")
                    _refiner_pipe = pipe
                    _refiner_device = "cpu"
                    log_event({"event": "RefinerLoad", "status": "cpu_fallback",
                               "path": REFINER_PATH})
                    print("[Refiner] CPU fallback ready.", flush=True)
                    return _refiner_pipe
                except Exception as e2:
                    log_event({"event": "RefinerLoadFailedCPU",
                               "error": str(e2),
                               "trace": traceback.format_exc()})
            _refiner_pipe = object()
            _refiner_device = "stub"
            return _refiner_pipe

def log_model_config(pipe) -> None:
    try:
        snap = _serialize_config(pipe)
        os.makedirs(SAVE_PATH, exist_ok=True)
        path = os.path.join(SAVE_PATH, "refiner_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False, default=str)
        log_event({"event": "RefinerConfigSaved", "path": path})
    except Exception as e:
        log_event({"event": "RefinerConfigWriteError", "error": str(e),
                   "trace": traceback.format_exc()})

def get_refiner_status() -> Dict[str, Any]:
    return {
        "available": REFINER_AVAILABLE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "path": REFINER_PATH,
        "importable": REFINER_IMPORTABLE,
        "timestamp": datetime.now().isoformat()
    }

def refine_image(
    base_image: Image.Image,
    prompt: str = PROMPT,
    negative: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    strength: float = 0.3,
    save: bool = True,
    save_path: str = SAVE_PATH,
    filename: Optional[str] = None,
    device: str = "cuda",
    fail_silent: bool = True
):
    if filename is None:
        filename = f"refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    # Accept either PIL.Image or QImage (convert if needed)
    try:
        from PySide6 import QtGui  # type: ignore
        if isinstance(base_image, QtGui.QImage):
            base_image = _qimage_to_pil(base_image)
    except Exception:
        pass

    if not isinstance(base_image, Image.Image):
        # Last fallback: cannot refine
        log_event({"event": "RefinerInputTypeUnsupported", "type": str(type(base_image))})
        return {"image": None, "path": None, "prompt": prompt,
                "device": "stub", "duration": 0.0, "filename": filename,
                "error": "Unsupported image type"}

    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")

    try:
        pipe = load_refiner(device=device, force_reload=False)
    except Exception as e:
        if not fail_silent: raise
        return {"image": base_image, "path": None, "prompt": prompt,
                "device": "stub", "duration": 0.0, "filename": filename,
                "error": str(e)}

    if not REFINER_AVAILABLE or _refiner_device == "stub" or not isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
        out_file = None
        if save:
            try:
                os.makedirs(save_path, exist_ok=True)
                out_file = os.path.join(save_path, filename)
                base_image.save(out_file)
            except Exception:
                out_file = None
        log_telemetry(event="RefinerFallback", prompt=prompt, device="stub",
                      filename=filename, saved=bool(out_file))
        return {"image": base_image, "path": out_file, "prompt": prompt,
                "device": "stub", "duration": 0.0, "filename": filename}

    start = time.time()
    resized = base_image
    if width and height:
        resized = base_image.resize((width, height), Image.LANCZOS)
        log_event({"event": "RefinerResize", "requested": (width, height),
                   "resized_to": resized.size})
    try:
        out = pipe(prompt=prompt, image=resized, strength=strength,
                   num_inference_steps=20, negative_prompt=negative)
        refined = out.images[0]
    except Exception as e:
        log_event({"event": "RefinerError", "error": str(e),
                   "trace": traceback.format_exc()})
        if not fail_silent: raise
        return {"image": base_image, "path": None, "prompt": prompt,
                "device": _refiner_device or "cuda", "duration": 0.0,
                "filename": filename, "error": str(e)}

    duration = round(time.time() - start, 2)
    log_memory(device)

    out_file = None
    if save:
        try:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, filename)
            refined.save(out_file)
        except Exception as e:
            log_event({"event": "RefinerSaveFailed", "error": str(e),
                       "trace": traceback.format_exc()})
            out_file = None

    log_telemetry(event="Refiner", duration=duration, prompt=prompt,
                  device=_refiner_device, filename=filename,
                  output_size=getattr(refined, "size", None),
                  path=out_file, saved=bool(out_file))

    return {"image": refined, "path": out_file, "prompt": prompt,
            "device": _refiner_device, "duration": duration,
            "filename": filename}

# -------- Helpers (added) --------
def _qimage_to_pil(qimg):
    """
    Convert a QtGui.QImage to a PIL.Image (RGB).
    """
    try:
        from PySide6 import QtGui  # type: ignore
        if qimg.format() != QtGui.QImage.Format.Format_RGBA8888:
            qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        width, height = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        buf = bytes(ptr)
        img = Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1)
        return img.convert("RGB")
    except Exception:
        raise

__all__ = ["load_refiner", "refine_image", "get_refiner_status",
           "log_event", "log_telemetry"]
