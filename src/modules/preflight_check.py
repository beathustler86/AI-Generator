from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import importlib, os

from src.modules.utils.telemetry import log_event  # near top (guarded try/except if preferred)

ROOT = Path(__file__).resolve().parent.parent  # project root
CORE_MANIFEST_PATH = ROOT / "config" / "core_manifest.json"
MODULE_MANIFEST_PATH = ROOT / "config" / "module_manifest.json"

# Updated paths to reflect actual locations under src/
FALLBACK_CORE = [
    "src/launch_gui.py",
    "src/gui/main_window.py",
    "src/modules/preflight_check.py",
    "src/modules/refiner_module.py",
    "src/modules/utils/telemetry.py",
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

def _check_gui_binding() -> dict:
    """Return which (if any) Qt bindings are importable."""
    result = {"PySide6": False, "PyQt5": False}
    for name in result.keys():
        try:
            __import__(name)
            result[name] = True
        except Exception:
            pass
    return result

def _check_xformers():
    want = os.getenv("ENABLE_XFORMERS","1") == "1"
    spec = importlib.util.find_spec("xformers")
    return {
        "want_xformers": want,
        "xformers_present": bool(spec)
    }

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

        gui = _check_gui_binding()
        if not any(gui.values()):
            missing_required.append("Qt binding (PySide6 or PyQt5) not installed")

        xformers_info = _check_xformers()
        result = {
            "status": "complete" if not missing_required else "failed",
            "missing_files": missing_required + missing_optional,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "core_count": len(core),
            "optional_count": len(optional),
            "auto_created_optional": auto_created,
            "qt_bindings": gui,
            "timestamp": datetime.utcnow().isoformat(),
            "xformers": xformers_info,
        }
        if xformers_info["want_xformers"] and not xformers_info["xformers_present"]:
            result.setdefault("missing_optional", []).append("xformers")
        
        log_event({"event": "preflight", "status": result["status"],
           "missing_required": len(missing_required),
           "missing_optional": len(missing_optional)})
        return result
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
            "qt_bindings": {},
            "timestamp": datetime.utcnow().isoformat(),
            "xformers": {},
        }

__all__ = ["run_preflight"]

if __name__ == "__main__":
    import pprint
    pprint.pp(run_preflight())
