from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CORE_MANIFEST_PATH = ROOT / "config" / "core_manifest.json"
MODULE_MANIFEST_PATH = ROOT / "config" / "module_manifest.json"

FALLBACK_CORE = [
    "launch_gui.py",
    "gui/main_window.py",
    "modules/preflight_check.py",
    "modules/refiner_module.py",
    "modules/utils/telemetry.py",
]

FALLBACK_OPTIONAL = [
    "outputs/refined/refiner_config.json",
    "outputs/images/refined/refiner_config.json",
]

OPTIONAL_AUTO_CREATE = {
    "outputs/refined/refiner_config.json": {"placeholder": True, "role": "refiner-global"},
    "outputs/images/refined/refiner_config.json": {"placeholder": True, "role": "refiner-images"},
}

def _safe_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}")
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii','ignore').decode('ascii')}")

def _load_manifest(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            files = data.get("files", [])
        elif isinstance(data, list):
            files = data
        else:
            return fallback
        return [str(f) for f in files if isinstance(f, str)]
    except Exception:
        return fallback

def _normalize(rel: str) -> str:
    return rel.replace("\\", "/").lstrip("/")

def run_preflight() -> dict:
    try:
        _safe_log("Preflight: verifying model checkpoints...")
        core = [_normalize(p) for p in _load_manifest(CORE_MANIFEST_PATH, FALLBACK_CORE)]
        mod_opt = [_normalize(p) for p in _load_manifest(MODULE_MANIFEST_PATH, FALLBACK_OPTIONAL)]
        optional = [p for p in mod_opt if p not in core]

        missing_required, missing_optional, auto_created = [], [], []

        for rel in core:
            if not (ROOT / rel).exists():
                missing_required.append(str((ROOT / rel).resolve()))

        for rel in optional:
            tgt = ROOT / rel
            if not tgt.exists():
                if rel in OPTIONAL_AUTO_CREATE:
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        tgt.write_text(
                            json.dumps(
                                {**OPTIONAL_AUTO_CREATE[rel], "created": True, "ts": datetime.utcnow().isoformat()},
                                ensure_ascii=False,
                                indent=2
                            ),
                            encoding="utf-8"
                        )
                        auto_created.append(str(tgt.resolve()))
                        continue
                    except Exception:
                        pass
                missing_optional.append(str(tgt.resolve()))

        status = "complete" if not missing_required else "failed"
        return {
            "status": status,
            "missing_files": missing_required + missing_optional,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "core_count": len(core),
            "optional_count": len(optional),
            "auto_created_optional": auto_created,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "missing_files": [],
            "missing_required": [],
            "missing_optional": [],
            "core_count": 0,
            "optional_count": 0,
            "auto_created_optional": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

__all__ = ["run_preflight"]

if __name__ == "__main__":
    import pprint
    pprint.pp(run_preflight())

def Get_TorchInfo(PythonExe):
    code = """
import json, os, glob, time
try:
    import torch
except Exception as e:
    print(json.dumps({"error":"torch import failed","exc":str(e)}))
    raise SystemExit(0)

info = {
  "torch_version": torch.__version__,
  "cuda_available": torch.cuda.is_available(),
  "cuda_build": getattr(getattr(torch,"version",None),"cuda",None),
  "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
  "cudnn_version": getattr(getattr(torch.backends,"cudnn",None),"version",lambda:None)(),
  "cudnn_enabled": getattr(getattr(torch.backends,"cudnn",None),"is_available",lambda:False)(),
}

if info["cuda_available"] and info["device_count"]>0:
    info["devices"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
    # simple timing
    a = torch.randn(4096,4096, device="cuda")
    b = torch.randn(4096,4096, device="cuda")
    torch.cuda.synchronize()
    t0=time.time()
    c = a@b
    torch.cuda.synchronize()
    info["matmul_ms"] = (time.time()-t0)*1000
else:
    info["devices"] = []

lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
patterns = ["cudnn*64_9*.dll","cudnn*64_8*.dll","cudnn*.dll"]
libs=[]
for pat in patterns:
    libs.extend(glob.glob(os.path.join(lib_dir, pat)))
info["cudnn_libs_found"] = sorted(set(libs))
print(json.dumps(info, indent=2))
"""
    import subprocess
    subprocess.run([PythonExe, "-c", code], check=True)

    code = """
import torch, torch.nn as nn
print("Device count:", torch.cuda.device_count(), "Current:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
x = torch.randn(1,3,128,128, device="cuda")
conv = nn.Conv2d(3,16,3).cuda()
y = conv(x)
print("Conv ok, y shape:", y.shape)
print("cuDNN enabled:", torch.backends.cudnn.is_available(), "version:", torch.backends.cudnn.version())
"""
    subprocess.run([PythonExe, "-c", code], check=True)