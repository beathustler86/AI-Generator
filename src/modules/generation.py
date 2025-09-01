from __future__ import annotations
import os, time, math
import json
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
import shutil
import torch
import hashlib
import multiprocessing as mp

from diffusers import (
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
)

# ---------------- Configuration / Environment ----------------
ENV_KEY = "MODEL_ID_OR_PATH"

DEFAULT_CACHE_ROOT = Path(os.environ.get(
    "PIPELINE_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "pipeline_cache")
)).resolve()

MODELS_ROOT      = Path(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models")
TEXT_TO_IMAGE_DIR = MODELS_ROOT / "text_to_image"
TEXT_TO_VIDEO_DIR = MODELS_ROOT / "text_to_video"

PREFERRED_LOCAL_NAMES = [
    "dreamshaper-xl-v2-turbo",
    "sdxl-base-1.0",
    "sdxl-base-1.5",
]

HF_FALLBACK = "stabilityai/stable-diffusion-xl-base-1.0"

_DISK_CACHE_ENABLED  = os.environ.get("DISK_PIPELINE_CACHE", "1") == "1"  # runtime mutable
DISK_CACHE_REBUILD  = os.environ.get("REBUILD_PIPELINE_CACHE", "0") == "1"
PIPELINE_DISK_CACHE_DIR = DEFAULT_CACHE_ROOT
PIPELINE_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_CACHE_MAX_ITEMS = int(os.environ.get("PIPELINE_CACHE_MAX_ITEMS", "6"))
PIPELINE_CACHE_MAX_GB    = float(os.environ.get("PIPELINE_CACHE_MAX_GB", "18.0"))

TORCH_COMPILE_ENABLED    = os.environ.get("TORCH_COMPILE", "0") == "1"
TORCH_COMPILE_MODE       = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_BACKEND    = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

PREFETCH_MODELS          = os.environ.get("PREFETCH_MODELS", "0") == "1"

# ---------------- In-memory pipeline cache ----------------
_PIPELINE = None
_PIPELINE_MODEL_ID: Optional[str] = None
_DEVICE = None
_PIPELINE_CACHE: Dict[tuple, object] = {}
_PIPELINE_CACHE_ORDER: List[tuple] = []
_PIPELINE_CACHE_MAX = 3  # in-RAM pipeline objects

LAST_SCHEDULER_NAME: Optional[str] = None

# Shared text encoder pool { sha256(file list hash) : module }
_SHARED_TEXT_ENCODERS: Dict[str, torch.nn.Module] = {}

_SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "dpm-sde": DPMSolverSDEScheduler,
}

_LOCK = threading.RLock()

# ---------------- Utility helpers ----------------
def _log_event(rec: Dict):
    try:
        from src.modules.utils.telemetry import log_event
        log_event(rec)
    except Exception:
        pass

def _is_diffusers_dir(p: Path) -> bool:
    return p.is_dir() and (p / "model_index.json").exists()

def _has_text_encoder_2(p: Path) -> bool:
    return (p / "text_encoder_2").exists() or (p / "tokenizer_2").exists()

def _similar_dirs(parent: Path, target_name: str) -> list[Path]:
    if not parent.exists(): return []
    tn = target_name.lower()
    out = []
    for d in parent.iterdir():
        if not d.is_dir(): continue
        name = d.name.lower()
        if name == tn or name.startswith(tn) or tn.startswith(name) or tn.replace("-", "") in name.replace("-", ""):
            out.append(d)
    return out

def _friendly_snapshot_label(p: Path) -> str:
    for comp in p.parents:
        if comp.name.startswith("models--"):
            segs = comp.name.split("--")
            if len(segs) >= 3:
                return segs[-1]
    return p.parent.name if len(p.name) >= 32 else p.name

def _friendly_label(path: str) -> str:
    p = Path(path)
    if "snapshots" in p.parts:
        return _friendly_snapshot_label(p)
    name = p.name
    if name.startswith("models--"):
        return name.split("--")[-1]
    return name

def list_local_image_models() -> list[str]:
    paths: list[str] = []
    if TEXT_TO_IMAGE_DIR.is_dir():
        for d in TEXT_TO_IMAGE_DIR.iterdir():
            if _is_diffusers_dir(d):
                paths.append(str(d.resolve()))
    for d in MODELS_ROOT.glob("models--*"):
        snap_root = d / "snapshots"
        if snap_root.is_dir():
            for s in snap_root.iterdir():
                if _is_diffusers_dir(s):
                    paths.append(str(s.resolve()))
    return sorted(paths)

def _video_model_entries() -> list[Tuple[str,str]]:
    entries: list[Tuple[str,str]] = []
    comfy = TEXT_TO_VIDEO_DIR / "ComfyUI"
    if comfy.exists():
        entries.append(("ComfyUI (video)", str(comfy.resolve())))
    for d in TEXT_TO_VIDEO_DIR.iterdir() if TEXT_TO_VIDEO_DIR.exists() else []:
        if d.is_dir() and d.name.lower().startswith("cosmos") and d != comfy:
            entries.append((f"{d.name} (video)", str(d.resolve())))
    return entries

def list_all_models() -> list[Dict]:
    out: list[Dict] = []
    for p in list_local_image_models():
        out.append({"label": _friendly_label(p), "path": p, "kind": "image"})
    existing = {e["label"] for e in out}
    for label, path in _video_model_entries():
        if label in existing: label += " (video)"
        out.append({"label": label, "path": path, "kind": "video"})
    seen: Dict[str,int] = {}
    for e in out:
        lbl = e["label"]
        if lbl in seen:
            seen[lbl] += 1
            e["label"] = f"{lbl}#{seen[lbl]}"
        else:
            seen[lbl] = 1
    return out

# Upscalers
def list_upscalers(root: Path = MODELS_ROOT) -> list[dict]:
    up_dir = root / "Upscaler"
    entries = []
    if up_dir.exists():
        for p in up_dir.rglob("*.pth"):
            if p.name.lower().startswith((".", "test")):
                continue
            entries.append({"name": p.stem, "path": str(p.resolve())})
    return sorted(entries, key=lambda x: x["name"])

def current_model_target() -> str:
    return os.environ.get(ENV_KEY, "")

def set_model_target(target: str):
    global _PIPELINE, _PIPELINE_MODEL_ID
    with _LOCK:
        _PIPELINE = None
        _PIPELINE_MODEL_ID = None
    os.environ[ENV_KEY] = target

def is_video_path(path_or_id: str) -> bool:
    p = path_or_id.lower()
    return "comfyui" in p or "text_to_video" in p or p.endswith("(video)")

def _classify_target() -> Tuple[str, bool]:
    env_val = os.environ.get(ENV_KEY, "").strip()
    if env_val:
        p = Path(env_val)
        if p.is_dir():
            if _is_diffusers_dir(p):
                return str(p), True
            cands = _similar_dirs(p.parent, p.name)
            if p.parent != TEXT_TO_IMAGE_DIR:
                cands += _similar_dirs(TEXT_TO_IMAGE_DIR, p.name)
            for c in cands:
                if _is_diffusers_dir(c):
                    print(f"[Generation] Auto-corrected model path '{p}' -> '{c}'", flush=True)
                    return str(c), True
            raise RuntimeError(f"Path '{p}' not a diffusers model (no model_index.json). Candidates: {[c.name for c in cands]}")
        else:
            base = Path(env_val).name
            cands = _similar_dirs(TEXT_TO_IMAGE_DIR, base)
            if cands:
                best = max(cands, key=lambda x: len(x.name))
                print(f"[Generation] Auto-corrected missing path '{env_val}' -> '{best}'", flush=True)
                return str(best), True
            return env_val.replace("\\", "/"), False
    locals_all = list_local_image_models()
    by_name = {Path(p).name: p for p in locals_all}
    for name in PREFERRED_LOCAL_NAMES:
        if name in by_name:
            return by_name[name], True
    if locals_all:
        return locals_all[0], True
    return HF_FALLBACK, False

def _select_pipeline_class(model_dir: Path):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    if _has_text_encoder_2(model_dir):
        return StableDiffusionXLPipeline
    return StableDiffusionPipeline

def _apply_scheduler(pipe, sampler: str):
    global LAST_SCHEDULER_NAME
    if not sampler: return pipe
    sampler = sampler.lower()
    if sampler == LAST_SCHEDULER_NAME:
        return pipe
    if sampler not in _SCHEDULER_MAP:
        return pipe
    cls = _SCHEDULER_MAP[sampler]
    try:
        new_sched = cls.from_config(pipe.scheduler.config)
        pipe.scheduler = new_sched
        LAST_SCHEDULER_NAME = sampler
    except Exception as e:
        print(f"[Generation] Scheduler swap failed ({sampler}): {e}", flush=True)
    return pipe

# ---------------- Disk cache (persisted) ----------------
def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]

def _canonical_path(p: str) -> str:
    try:
        cp = str(Path(p).resolve())
    except Exception:
        cp = p
    # Normalize case on Windows so mapping matches
    if os.name == "nt":
        cp = os.path.normcase(cp)
    return cp.replace("\\", "/")

def _cache_slug_for(path: str) -> str:
    path = _canonical_path(path)
    p = Path(path)
    base = p.name if p.exists() else Path(path).name
    if len(base) > 48:
        base = base[:48]
    return f"{base}-{_hash_key(path)}"

def _resolve_cached_path(original: str) -> Optional[str]:
    if not _DISK_CACHE_ENABLED or DISK_CACHE_REBUILD:
        return None
    orig_canon = _canonical_path(original)
    slug = _cache_slug_for(orig_canon)
    cdir = PIPELINE_DISK_CACHE_DIR / slug
    if (cdir / "model_index.json").exists():
        _log_event({"event":"disk_cache_hit","slug":slug})
        return str(cdir.resolve())
    # Mapping file shortcut
    map_file = PIPELINE_DISK_CACHE_DIR / "mapping.json"
    try:
        if map_file.exists():
            mapping = json.loads(map_file.read_text(encoding="utf-8"))
            slug = mapping.get(orig_canon) or mapping.get(original)
            if slug:
                cdir = PIPELINE_DISK_CACHE_DIR / slug
                if (cdir / "model_index.json").exists():
                    _log_event({"event":"disk_cache_hit","slug":slug,"mapped":True})
                    return str(cdir.resolve())
    except Exception:
        pass
    _log_event({"event":"disk_cache_miss","original":original})
    return None

def _write_disk_pipeline_cache(original: str, pipe) -> None:
    if not _DISK_CACHE_ENABLED:
        return
    try:
        original_canon = _canonical_path(original)
        slug = _cache_slug_for(original_canon)
        cdir = PIPELINE_DISK_CACHE_DIR / slug
        if cdir.exists() and not DISK_CACHE_REBUILD:
            return
        if cdir.exists() and DISK_CACHE_REBUILD:
            shutil.rmtree(cdir, ignore_errors=True)
        print(f"[Generation] Writing disk pipeline cache: {cdir}", flush=True)
        pipe.save_pretrained(str(cdir), safe_serialization=True)
        _log_event({"event":"disk_cache_write","slug":slug})
        _prune_disk_cache()
        # Update mapping (dedup)
        try:
            map_file = PIPELINE_DISK_CACHE_DIR / "mapping.json"
            mapping = {}
            if map_file.exists():
                mapping = json.loads(map_file.read_text(encoding="utf-8"))
            mapping[original_canon] = slug
            map_file.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        except Exception:
            pass
    except Exception as e:
        print(f"[Generation] Disk cache save failed: {e}", flush=True)

def _disk_cache_entries() -> List[Path]:
    if not PIPELINE_DISK_CACHE_DIR.exists():
        return []
    return [p for p in PIPELINE_DISK_CACHE_DIR.iterdir() if (p / "model_index.json").exists()]

def _prune_disk_cache():
    entries = _disk_cache_entries()
    if not entries:
        return
    # LRU by modified time
    entries.sort(key=lambda p: p.stat().st_mtime)
    # Size-based prune
    total_bytes = sum(_dir_size(e) for e in entries)
    max_bytes = PIPELINE_CACHE_MAX_GB * (1024**3)
    pruned = False
    while (len(entries) > PIPELINE_CACHE_MAX_ITEMS) or (total_bytes > max_bytes and len(entries)>1):
        victim = entries.pop(0)
        vsize = _dir_size(victim)
        try:
            shutil.rmtree(victim, ignore_errors=True)
            total_bytes -= vsize
            _log_event({"event":"disk_cache_prune","path":str(victim),"freed_bytes":vsize})
            pruned = True
        except Exception:
            pass
    if pruned:
        print("[Generation] Disk cache pruned.", flush=True)

def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try: total += p.stat().st_size
            except Exception: pass
    return total

# ---------------- Remediation (fp16 filename) ----------------
def _remap_diffusion_file(subdir_dir: Path, disable_env_flag: str):
    if not subdir_dir.is_dir(): return
    if os.environ.get(disable_env_flag) == "1": return
    sentinel = subdir_dir / ".remap_done"
    if sentinel.exists():
        return
    target = subdir_dir / "diffusion_pytorch_model.safetensors"
    if target.exists():
        sentinel.write_text("ok", encoding="utf-8")
        return
    candidates = []
    for pat in [
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model-fp16.safetensors",
        "diffusion_pytorch_model_fp16.safetensors",
        "*fp16*.safetensors",
        "*fp32*.safetensors",
    ]:
        candidates.extend(subdir_dir.glob(pat))
    cands = [c for c in candidates if c.is_file()]
    if not cands:
        return
    cands.sort(key=lambda p: (0 if "fp16" in p.name.lower() else 1, len(p.name)))
    src = cands[0]
    try:
        if hasattr(os, "link"):
            os.link(src, target)
        else:
            shutil.copyfile(src, target)
        print(f"[Generation] Remap {subdir_dir.name}: {src.name} -> {target.name}", flush=True)
        sentinel.write_text("ok", encoding="utf-8")
    except Exception as e:
        print(f"[Generation] Remap failed ({subdir_dir.name}): {e}", flush=True)

# Extend remediation patterns to include generic 'model.safetensors' alias in text_encoder dirs.
def _remap_generic_model(subdir: Path):
    """
    Ensure model.safetensors exists if only model.fp16.safetensors (or related) present.
    """
    if not subdir.is_dir(): return
    target = subdir / "model.safetensors"
    if target.exists(): return
    cands = []
    for pat in ["model.fp16.safetensors","model-fp16.safetensors","*fp16*.safetensors","model.sft","pytorch_model.safetensors"]:
        cands.extend(subdir.glob(pat))
    cands = [c for c in cands if c.is_file()]
    if not cands: return
    cands.sort(key=lambda p: (0 if "fp16" in p.name.lower() else 1, len(p.name)))
    src = cands[0]
    try:
        if hasattr(os, "link"):
            os.link(src, target)
        else:
            shutil.copyfile(src, target)
        print(f"[Generation] Remap generic model: {src.name} -> {target.name}", flush=True)
    except Exception as e:
        print(f"[Generation] Generic model remap failed: {e}", flush=True)

def _cold_load_remediate(model_root: Path):
    fast = os.environ.get("FAST_DIFFUSERS_LOAD") == "1"
    if fast:
        vae_ok = (model_root / "vae" / "diffusion_pytorch_model.safetensors").exists()
        unet_ok = (model_root / "unet" / "diffusion_pytorch_model.safetensors").exists()
        if vae_ok and unet_ok:
            return
    _remap_diffusion_file(model_root / "vae", "DISABLE_VAE_FP16_REMAP")
    _remap_diffusion_file(model_root / "unet", "DISABLE_UNET_FP16_REMAP")
    _remap_generic_model(model_root / "text_encoder")
    _remap_generic_model(model_root / "text_encoder_2")

# ---------------- Shared text encoder reuse ----------------
def _collect_te_state_dirs(root: Path) -> List[Path]:
    out=[]
    for name in ("text_encoder","text_encoder_2"):
        d = root / name
        if d.exists(): out.append(d)
    return out

def _hash_dir_files(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*.safetensors")):
        try:
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
        except Exception:
            pass
    return h.hexdigest()[:16]

def _maybe_reuse_text_encoders(root: Path, pipe):
    try:
        dirs = _collect_te_state_dirs(root)
        for d in dirs:
            sig = _hash_dir_files(d)
            if not sig:
                continue
            if sig in _SHARED_TEXT_ENCODERS:
                # swap in existing module (attribute names vary)
                attr = "text_encoder_2" if "text_encoder_2" in d.name else "text_encoder"
                # Some pipelines use pipe.text_encoder / pipe.text_encoder_2
                if hasattr(pipe, attr):
                    setattr(pipe, attr, _SHARED_TEXT_ENCODERS[sig])
                    _log_event({"event":"text_encoder_reuse","which":attr,"sig":sig})
            else:
                # Register new
                attr = "text_encoder_2" if d.name == "text_encoder_2" else "text_encoder"
                mod = getattr(pipe, attr, None)
                if mod is not None:
                    _SHARED_TEXT_ENCODERS[sig] = mod
    except Exception:
        pass

# ---------------- Pipeline loader ----------------
def _load_pipeline(device: str = "cuda", sampler: str = ""):
    global _PIPELINE, _PIPELINE_MODEL_ID, _DEVICE, _PIPELINE_CACHE
    with _LOCK:
        target, is_local = _classify_target()
        if is_video_path(target):
            raise RuntimeError("Selected video model; image pipeline not applicable.")
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        use_cuda = (device == "cuda" and torch.cuda.is_available())
        model_path = Path(target) if is_local else None

        cached_override = _resolve_cached_path(target) if is_local else None
        if cached_override:
            target = cached_override
            model_path = Path(target)
            is_local = True
            print(f"[Generation] Using disk-cached pipeline: {target}", flush=True)

        pipeline_cls = _select_pipeline_class(model_path) if (is_local and model_path) else (
            StableDiffusionXLPipeline if "xl" in target.lower() else StableDiffusionPipeline
        )
        cache_key = (target, 'cuda' if use_cuda else 'cpu')
        if cache_key in _PIPELINE_CACHE:
            pipe = _PIPELINE_CACHE[cache_key]
            _PIPELINE = pipe
            _PIPELINE_MODEL_ID = target
            _DEVICE = cache_key[1]
            if cache_key in _PIPELINE_CACHE_ORDER:
                _PIPELINE_CACHE_ORDER.remove(cache_key)
            _PIPELINE_CACHE_ORDER.append(cache_key)
            _apply_scheduler(pipe, sampler)
            return _PIPELINE

        t0 = time.time()
        print(f"[Generation] Cold load start: {target}", flush=True)

        if is_local and model_path and not str(model_path).startswith(str(PIPELINE_DISK_CACHE_DIR)):
            _cold_load_remediate(model_path)

        dtype = torch.float16 if use_cuda else torch.float32
        load_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=os.environ.get("DIFFUSERS_FORCE_LOCAL_FILES") == "1",
            low_cpu_mem_usage=True,
        )
        if pipeline_cls.__name__.endswith("XLPipeline"):
            load_kwargs["safety_checker"] = None

        # Fallback logic: if text_encoder_2 missing, drop it from components
        try:
            pipe = pipeline_cls.from_pretrained(target, **load_kwargs)
        except ValueError as e:
            print(f"[Generation] Retry minimal load due to error: {e}", flush=True)
            load_kwargs = dict(torch_dtype=dtype)
            if pipeline_cls.__name__.endswith("XLPipeline"):
                load_kwargs["safety_checker"] = None
            pipe = pipeline_cls.from_pretrained(target, **load_kwargs)
        except OSError as e:
            msg = str(e).lower()
            if "text_encoder_2" in msg or "model.safetensors" in msg:
                print("[Generation] Missing second text encoder – retry lite SDXL (single encoder).", flush=True)
                # Force pipeline to treat as non-XL if second encoder missing
                from diffusers import StableDiffusionPipeline
                pipeline_cls = StableDiffusionPipeline
                lite_kwargs = dict(torch_dtype=dtype, use_safetensors=True, local_files_only=load_kwargs.get("local_files_only", False), low_cpu_mem_usage=True, safety_checker=None)
                pipe = pipeline_cls.from_pretrained(target, **lite_kwargs)
                _log_event({"event":"sdxl_lite_mode","model":target})
            else:
                raise

        if use_cuda:
            pipe.to("cuda")

        # Shared text encoders (reuse)
        if is_local and model_path:
            _maybe_reuse_text_encoders(model_path, pipe)

        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3) if use_cuda else 0
        except Exception:
            gpu_mem = 0
        if use_cuda and gpu_mem <= 10:
            if hasattr(pipe, "enable_attention_slicing"):
                try: pipe.enable_attention_slicing()
                except Exception: pass
        if use_cuda and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception: pass

        _apply_scheduler(pipe, sampler)

        _PIPELINE_CACHE[cache_key] = pipe
        _PIPELINE_CACHE_ORDER.append(cache_key)
        while len(_PIPELINE_CACHE_ORDER) > _PIPELINE_CACHE_MAX:
            old = _PIPELINE_CACHE_ORDER.pop(0)
            if old == cache_key: continue
            old_pipe = _PIPELINE_CACHE.pop(old, None)
            try:
                if hasattr(old_pipe, "to"):
                    old_pipe.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        _PIPELINE = pipe
        _PIPELINE_MODEL_ID = target
        _DEVICE = cache_key[1]
        load_ms = int((time.time() - t0)*1000)
        _log_event({"event":"model_load_time","model":target,"ms":load_ms})
        print(f"[Generation] Cold load done in {load_ms} ms: {target}", flush=True)

        # Write disk cache if not already cached
        if _DISK_CACHE_ENABLED and is_local and model_path and not str(model_path).startswith(str(PIPELINE_DISK_CACHE_DIR)):
            _write_disk_pipeline_cache(original_target, pipe)
            # Persist mapping so next process start can directly use cache (write mapping file)
            try:
                map_file = PIPELINE_DISK_CACHE_DIR / "mapping.json"
                mapping = {}
                if map_file.exists():
                    mapping = json.loads(map_file.read_text(encoding="utf-8"))
                slug = _cache_slug_for(model_path.as_posix())
                mapping[model_path.as_posix()] = slug
                map_file.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
            except Exception:
                pass
        return _PIPELINE

# ---------------- Torch compile (async) ----------------
def _async_compile_unet(pipe):
    if not hasattr(pipe, "unet"):
        return
    def _run():
        try:
            _log_event({"event":"torch_compile_start"})
            unet = pipe.unet
            # Warm dummy forward (small)
            with torch.inference_mode():
                sample = torch.randn(1, unet.in_channels, 64, 64, generator=torch.Generator(device=unet.device))
                # Some UNets need timestep & encoder hidden states; attempt generic call
                try:
                    if "timestep" in unet.forward.__code__.co_varnames:
                        unet(sample, 0.0)
                    else:
                        unet(sample)
                except Exception:
                    pass
            compiled = torch.compile(unet, mode=TORCH_COMPILE_MODE, backend=TORCH_COMPILE_BACKEND)
            pipe.unet = compiled
            _log_event({"event":"torch_compile_done"})
            print("[Generation] UNet torch.compile complete.", flush=True)
        except Exception as e:
            print(f"[Generation] torch.compile skipped: {e}", flush=True)
    threading.Thread(target=_run, daemon=True).start()

# ---------------- Prefetch (multiprocess) ----------------
def _prefetch_worker(model_paths: List[str]):
    os.environ["DISK_PIPELINE_CACHE"] = "1"
    for p in model_paths:
        try:
            os.environ[ENV_KEY] = p
            _ = _load_pipeline(device="cuda" if torch.cuda.is_available() else "cpu", sampler="euler_a")
        except Exception:
            pass

def start_prefetch():
    if not PREFETCH_MODELS:
        return
    imgs = list_local_image_models()
    # Exclude currently loaded
    current = current_model_target()
    remaining = [p for p in imgs if p != current][:4]  # cap
    if not remaining:
        return
    _log_event({"event":"prefetch_start","count":len(remaining)})
    proc = mp.Process(target=_prefetch_worker, args=(remaining,), daemon=True)
    proc.start()

# ---------------- Public generation functions ----------------
def generate_images(
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    batch: int,
    sampler: str,
    device: str = "cuda",
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> List["object"]:
    pipe = _load_pipeline(device=device, sampler=sampler)
    def _round8(x: int) -> int: return max(64, (x // 8) * 8)
    width, height = _round8(width), _round8(height)
    images = []
    gen_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    def _callback(step: int, timestep: int, latents):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if progress_cb and steps > 0 and (step == 0 or step == steps - 1 or (step % max(1, steps // 10) == 0)):
            progress_cb(f"Sampling step {step}/{steps}")
    for i in range(batch):
        if cancel_event and cancel_event.is_set(): break
        g = torch.Generator(device=gen_device).manual_seed(seed + i)
        with torch.inference_mode():
            out = pipe(prompt=prompt,
                       negative_prompt=negative or None,
                       guidance_scale=cfg,
                       num_inference_steps=steps,
                       width=width,
                       height=height,
                       generator=g,
                       callback=_callback,
                       callback_steps=1)
        images.append(out.images[0])
    return images

def generate_video(
    prompt: str,
    frames: int = 16,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    fps: int = 8,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    negative: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> List["object"]:
    try:
        from src.nodes.cosmos_backend import generate_frames as cosmos_frames
        return cosmos_frames(prompt, frames, width, height,
                             seed, fps, progress_cb,
                             cancel_event=cancel_event,
                             steps=steps, cfg=cfg, negative=negative)
    except Exception as e:
        if progress_cb: progress_cb(f"Video backend fatal: {e}")
        from PIL import Image
        return [Image.new("RGB",(width,height),(0,0,0)) for _ in range(frames)]

def pil_to_qimage(pil_img):
    from PIL import Image  # noqa
    from PySide6 import QtGui  # type: ignore
    if pil_img.mode not in ("RGB","RGBA"):
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", pil_img.mode)
    fmt = QtGui.QImage.Format.Format_RGBA8888 if pil_img.mode == "RGBA" else QtGui.QImage.Format.Format_RGB888
    return QtGui.QImage(data, pil_img.width, pil_img.height, fmt).copy()

# Optional: expose prefetch trigger
def trigger_background_prefetch():
    try:
        start_prefetch()
    except Exception:
        pass

# -------- Runtime toggles --------
def set_disk_cache_enabled(enabled: bool):
    global _DISK_CACHE_ENABLED
    _DISK_CACHE_ENABLED = bool(enabled)
    _log_event({"event":"pipeline_cache_toggle","enabled":_DISK_CACHE_ENABLED})

def is_disk_cache_enabled() -> bool:
    return _DISK_CACHE_ENABLED

# -------- VRAM / pipeline release --------
def release_pipeline(free_ram: bool = False):
    global _PIPELINE, _PIPELINE_MODEL_ID, _DEVICE
    with _LOCK:
        if _PIPELINE is not None:
            try:
                if free_ram:
                    _PIPELINE.to("cpu")
                    torch.cuda.empty_cache()
                _PIPELINE = None
                _PIPELINE_MODEL_ID = None
                _DEVICE = None
            except Exception as e:
                print(f"[Generation] Pipeline release failed: {e}", flush=True)
                _log_event({"event":"pipeline_release_error","message":str(e)})
        
# -------- Cache diagnostics / purge --------
def disk_cache_root() -> str:
    return str(PIPELINE_DISK_CACHE_DIR)

def _entry_info(p: Path) -> dict:
    try:
        return {
            "slug": p.name,
            "files": sum(1 for _f in p.rglob("*") if _f.is_file()),
            "mb": round(_dir_size(p) / (1024*1024), 2),
            "age_s": int(time.time() - p.stat().st_mtime)
        }
    except Exception:
        return {"slug": p.name}

def get_cache_stats() -> dict:
    ents = _disk_cache_entries()
    info = [_entry_info(e) for e in ents]
    total_bytes = sum(_dir_size(e) for e in ents)
    data = {
        "root": disk_cache_root(),
        "enabled": _DISK_CACHE_ENABLED,
        "entries": info,
        "total_mb": round(total_bytes / (1024*1024), 2),
        "count": len(info)
    }
    _log_event({"event":"disk_cache_stats","count":data["count"],"total_mb":data["total_mb"]})
    return data

def purge_disk_cache():
    ents = _disk_cache_entries()
    freed = 0
    for e in ents:
        try:
            sz = _dir_size(e)
            shutil.rmtree(e, ignore_errors=True)
            freed += sz
        except Exception:
            pass
    map_file = PIPELINE_DISK_CACHE_DIR / "mapping.json"
    if map_file.exists():
        try: map_file.unlink()
        except Exception: pass
    _log_event({"event":"disk_cache_purge","freed_mb":round(freed/(1024*1024),2)})
    print(f"[Generation] Disk cache purged, freed ~{freed/(1024*1024):.2f} MB", flush=True)
