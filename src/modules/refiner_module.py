"""
Refiner subsystem — improved robustness:
- safe telemetry writer (ensures directories, JSON-safe)
- cached pipeline instance with thread lock to avoid repeated heavy loads
- CPU fallback on OOM or load failure
- safer config serialization
- best-effort fallback behavior on runtime refine errors
"""
from __future__ import annotations
import os
import json
import threading
import traceback
import time
from datetime import datetime
from PIL import Image
import torch
from pathlib import Path
from typing import Optional

# === CONFIG ===
REFINER_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-refiner-1.0"
SAVE_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/output/refined"
PROMPT = "cockpit-grade GUI with tactical overlays"

# Telemetry file (best-effort location)
_TELEMETRY_LOG = os.environ.get(
	"AI_GENERATOR_TELEMETRY",
	os.path.join("F:/SoftwareDevelopment/AI Models Image/AIGenerator/outputs", "logs", "telemetry_logs", "telemetry.jsonl")
)

# Attempt Import
try:
	from diffusers import StableDiffusionXLImg2ImgPipeline
	REFINER_IMPORTABLE = True
except Exception:
	StableDiffusionXLImg2ImgPipeline = None
	REFINER_IMPORTABLE = False

REFINER_AVAILABLE = os.path.isdir(REFINER_PATH) and REFINER_IMPORTABLE

# Cache + lock for pipeline reuse
_refiner_lock = threading.RLock()
_refiner_pipe = None
_refiner_device = None


def _safe_write_jsonl(path, payload):
	try:
		dirpath = os.path.dirname(path)
		if dirpath:
			os.makedirs(dirpath, exist_ok=True)
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
	except Exception:
		# last-resort: print to console but do not raise
		try:
			print(f"[TelemetryWriteFailed] {payload}")
		except Exception:
			pass


def log_event(event):
	"""Write an event (dict or string) to telemetry file — never raise."""
	if isinstance(event, dict):
		payload = event.copy()
		if "timestamp" not in payload:
			payload["timestamp"] = datetime.now().isoformat()
	else:
		payload = {"message": str(event), "timestamp": datetime.now().isoformat()}
	_safe_write_jsonl(_TELEMETRY_LOG, payload)


def log_telemetry(**kwargs):
	entry = {**kwargs}
	if "timestamp" not in entry:
		entry["timestamp"] = datetime.now().isoformat()
	log_event(entry)


def log_memory(device):
	try:
		if device == "cuda" and torch.cuda.is_available():
			mem = torch.cuda.memory_allocated() / 1024**2
			log_event({"event": "RefinerMemory", "device": device, "allocated_mb": round(mem, 2), "timestamp": datetime.now().isoformat()})
	except Exception:
		# Do not fail telemetry on memory fetch error
		pass


def _serialize_config(pipe):
	"""Return a JSON-serializable representation of pipe.config (best-effort)."""
	try:
		cfg = getattr(pipe, "config", None)
		if cfg is None:
			return {"config": None}
		# Many diffusers configs are dataclasses / dict-like — try dict() then fallback to str()
		try:
			return dict(cfg)
		except Exception:
			try:
				return json.loads(json.dumps(cfg, default=str))
			except Exception:
				return {"config_str": str(cfg)}
	except Exception:
		return {"config_error": "failed to serialize config"}


def load_refiner(device: str = "cuda", force_reload: bool = False):
    """
    Load and cache the refiner pipeline.
    - Reuses cached pipeline when possible.
    - Attempts CPU fallback on failure.
    - Returns StableDiffusionXLImg2ImgPipeline instance on success or raises RuntimeError.
    """
    global _refiner_pipe, _refiner_device

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not REFINER_AVAILABLE:
        log_event({"event": "RefinerUnavailable", "path": REFINER_PATH, "device": device, "timestamp": datetime.now().isoformat()})
        raise RuntimeError("Refiner pipeline not available.")

    with _refiner_lock:
        # return cached if matches device and not forced
        if _refiner_pipe is not None and not force_reload and _refiner_device == device:
            return _refiner_pipe

        # attempt to load on requested device
        try:
            print("[Refiner] Loading img2img model...", flush=True)
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                REFINER_PATH,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
            _refiner_pipe = pipe
            _refiner_device = device
            # log a lightweight config snapshot
            log_event({"event": "RefinerLoad", "status": "success", "device": device, "path": REFINER_PATH, "config": _serialize_config(pipe)})
            print("[Refiner] Ready.", flush=True)
            return _refiner_pipe
        except Exception as e:
            # on CUDA failures, try CPU fallback before giving up
            trace = traceback.format_exc()
            log_event({"event": "RefinerLoadFailed", "device": device, "error": str(e), "trace": trace})
            print(f"[Refiner] Load failed on {device}: {e}", flush=True)
            if device != "cpu":
                try:
                    print("[Refiner] Retrying load on CPU...", flush=True)
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        REFINER_PATH,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                    ).to("cpu")
                    _refiner_pipe = pipe
                    _refiner_device = "cpu"
                    log_event({"event": "RefinerLoad", "status": "cpu_fallback", "path": REFINER_PATH})
                    print("[Refiner] CPU fallback ready.", flush=True)
                    return _refiner_pipe
                except Exception as e2:
                    trace2 = traceback.format_exc()
                    log_event({"event": "RefinerLoadFailedCPU", "error": str(e2), "trace": trace2})
                    print(f"[Refiner] CPU fallback failed: {e2}", flush=True)
            raise RuntimeError(f"Failed to load refiner pipeline: {e}")


def log_model_config(pipe):
	"""Write a serializable excerpt of the pipeline config to outputs (best-effort)."""
	try:
		config_snapshot = _serialize_config(pipe)
		config_dir = os.path.join(SAVE_PATH)
		os.makedirs(config_dir, exist_ok=True)
		config_path = os.path.join(config_dir, "refiner_config.json")
		with open(config_path, "w", encoding="utf-8") as f:
			json.dump(config_snapshot, f, indent=2, ensure_ascii=False, default=str)
		log_event({"event": "RefinerConfigSaved", "path": config_path})
		print(f"[Refiner] ✅ Config written to: {config_path}", flush=True)
	except Exception as e:
		log_event({"event": "RefinerConfigWriteError", "error": str(e), "trace": traceback.format_exc()})
		print(f"[Refiner] ❌ Failed to write config: {e}", flush=True)


def get_refiner_status():
	return {
		"available": REFINER_AVAILABLE,
		"device": "cuda" if torch.cuda.is_available() else "cpu",
		"path": REFINER_PATH,
		"importable": REFINER_IMPORTABLE,
		"timestamp": datetime.now().isoformat()
	}


def refine_image(base_image, prompt=PROMPT, negative=None, width=None, height=None, strength=0.3, save=True, save_path=SAVE_PATH, filename=None, device="cuda", fail_silent=False):
	"""
	Refine base_image and return dict with results.
	- If pipeline unavailable or error occurs, returns fallback (and optionally raises unless fail_silent=True).
	- Caches pipeline to avoid repeated heavy loads.
	"""
	if filename is None:
		filename = f"refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

	if device == "cuda" and not torch.cuda.is_available():
		print("[Refiner] CUDA not available — switching to CPU.", flush=True)
		device = "cpu"

	# Fallback mode if refiner not available
	if not REFINER_AVAILABLE:
		print("⚠️ Refiner pipeline unavailable — using fallback.", flush=True)
		if save:
			try:
				os.makedirs(save_path, exist_ok=True)
				output_file = os.path.join(save_path, filename)
				base_image.save(output_file)
				print(f"[Fallback] Saved to {output_file}", flush=True)
			except Exception as e:
				output_file = None
				log_event({"event": "RefinerFallbackSaveFailed", "error": str(e), "trace": traceback.format_exc()})
		else:
			output_file = None

		log_telemetry(event="RefinerFallback", duration=0.0, prompt=prompt, device="stub", filename=filename, output_size=base_image.size, path=output_file, saved=save)
		return {"image": base_image, "path": output_file, "prompt": prompt, "device": "stub", "duration": 0.0, "filename": filename}

	# Try to load or reuse pipeline
	try:
		refiner = load_refiner(device)
		# persist a lightweight config
		log_model_config(refiner)
		log_event({"event": "RefinerLoadedRuntime", "path": REFINER_PATH, "device": device, "timestamp": datetime.now().isoformat()})
	except Exception as e:
		# loading failed — fallback
		trace = traceback.format_exc()
		log_event({"event": "RefinerRuntimeLoadFailed", "error": str(e), "trace": trace})
		print(f"[Refiner] Runtime load failed: {e}", flush=True)
		if fail_silent:
			# behave like fallback
			if save:
				try:
					os.makedirs(save_path, exist_ok=True)
					output_file = os.path.join(save_path, filename)
					base_image.save(output_file)
				except Exception as e2:
					output_file = None
					log_event({"event": "RefinerFallbackSaveFailed", "error": str(e2), "trace": traceback.format_exc()})
			else:
				output_file = None
			return {"image": base_image, "path": output_file, "prompt": prompt, "device": "stub", "duration": 0.0, "filename": filename, "error": str(e)}
		raise

	# Prepare image
	start = time.time()
	if base_image.mode != "RGB":
		base_image = base_image.convert("RGB")

	if width and height:
		target_size = (width, height)
		base_image = base_image.resize(target_size, Image.LANCZOS)
		print(f"[Refiner] Resized to GUI-specified resolution: {target_size}", flush=True)
	else:
		target_size = (base_image.width, base_image.height)
		print(f"[Refiner] Using original image resolution: {target_size}", flush=True)

	log_event({"event": "RefinerResize", "requested": (width, height), "resized_to": base_image.size, "timestamp": datetime.now().isoformat()})

	# Run refinement (capture errors; optionally fallback)
	try:
		out = refiner(prompt=prompt, image=base_image, strength=strength, num_inference_steps=20)
		refined = out.images[0]
	except Exception as e:
		trace = traceback.format_exc()
		log_event({"event": "RefinerError", "error": str(e), "trace": trace, "prompt": prompt, "device": device, "filename": filename})
		print(f"[Refiner] Error during refinement: {e}", flush=True)
		if fail_silent:
			# fallback save original
			output_file = None
			if save:
				try:
					os.makedirs(save_path, exist_ok=True)
					output_file = os.path.join(save_path, filename)
					base_image.save(output_file)
				except Exception as e2:
					log_event({"event": "RefinerFallbackSaveFailed", "error": str(e2), "trace": traceback.format_exc()})
			return {"image": base_image, "path": output_file, "prompt": prompt, "device": "stub", "duration": 0.0, "filename": filename, "error": str(e)}
		raise

	# Postprocess / save
	duration = round(time.time() - start, 2)
	log_memory(device)

	output_file = None
	if save:
		try:
			os.makedirs(save_path, exist_ok=True)
			output_file = os.path.join(save_path, filename)
			refined.save(output_file)
			print(f"[Refiner] Saved to {output_file}", flush=True)
		except Exception as e:
			log_event({"event": "RefinerSaveFailed", "error": str(e), "trace": traceback.format_exc()})
			output_file = None

	log_telemetry(event="Refiner", duration=duration, prompt=prompt, device=device, filename=filename, output_size=getattr(refined, "size", None), path=output_file, saved=bool(output_file), timestamp=datetime.now().isoformat())

	return {"image": refined, "path": output_file, "prompt": prompt, "device": device, "duration": duration, "filename": filename}


from __future__ import annotations
import torch
from pathlib import Path
from typing import Optional
from PIL import Image

_refiner = None
_refiner_device = None

def load_refiner(device: str = "cuda", force_reload: bool = False):
    global _refiner, _refiner_device
    if _refiner is not None and not force_reload:
        return _refiner
    # Placeholder: real implementation should load an actual refiner pipeline / model
    _refiner = object()
    _refiner_device = device
    return _refiner

def refine_image(pil_image: Image.Image) -> Image.Image:
    """
    Dummy passthrough refiner.
    Replace with real refinement logic.
    """
    return pil_image

