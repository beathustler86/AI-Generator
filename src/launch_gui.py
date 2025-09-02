from __future__ import annotations
from typing import Optional, Any, List
import os, sys, threading
from pathlib import Path
import traceback
import json
import importlib
import platform
import torch  # noqa

# Project paths
CURRENT = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT.parent
for p in (str(PROJECT_ROOT), str(CURRENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional: quiet torchvision shim
try:
    from src.compat.tv_functional_tensor_shim import ensure_installed as _shim_ensure
    _shim_ensure(verbose=os.getenv("SHIM_VERBOSE") in ("1","true","TRUE","on","On"))
except Exception as e:
    print(f"[Compat] Shim init failed: {e}", flush=True)

def _run_preflight():
    try:
        from src.modules.preflight_check import run_preflight
        result = run_preflight()
        (PROJECT_ROOT / "preflight_last.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[Preflight] status={result.get('status')} "
              f"missing_required={len(result.get('missing_required', []))} "
              f"missing_optional={len(result.get('missing_optional', []))} "
              f"auto_created={len(result.get('auto_created_optional', []))}", flush=True)
    except Exception as e:
        print(f"[Preflight] skipped: {e}", flush=True)

_run_preflight()

try:
    from PySide6 import QtWidgets  # type: ignore
except ImportError as e:
    print("[FATAL] Qt import failed:", e)
    raise SystemExit(1)

# Import after sys.path prepared
try:
    from src.gui.main_window import MainWindow
    from src.gui.ui_setup import Services
    from src.modules.model_manager import ModelManager
    import src.modules.generation as gen_mod
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

DEFAULT_MODELS_ROOT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models"

def _log_env_snapshot():
    try:
        from src.modules.utils.telemetry import log_event
        vers = {}
        for n in ("torch","diffusers","transformers","huggingface_hub","xformers"):
            try:
                m = importlib.import_module(n)
                vers[n] = getattr(m, "__version__", "?")
            except Exception:
                vers[n] = "missing"
        specs = {}
        try:
            specs["torch_cuda"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                specs["cuda_version"] = torch.version.cuda
                specs["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        log_event({"event":"env_snapshot","versions":vers,"specs":specs})
    except Exception:
        pass

def _auto_set_model(models_root):
    try:
        if not gen_mod.current_model_target():
            from src.modules.generation import list_local_image_models, set_model_target
            candidates = list_local_image_models()
            if candidates:
                set_model_target(candidates[0])
                print(f"[AutoModel] Default model set: {candidates[0]}", flush=True)
            else:
                print("[AutoModel] No local models under:", models_root, flush=True)
    except Exception as e:
        print(f"[AutoModel] Failed: {e}", flush=True)

def main():
    app = QtWidgets.QApplication(sys.argv)

    if os.getenv("DISABLE_ENV_SNAPSHOT","0") != "1":
        _log_env_snapshot()

    models_root = Path(os.getenv("MODELS_ROOT", DEFAULT_MODELS_ROOT))
    cfg_path = PROJECT_ROOT / "config" / "app.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            models_root = Path(cfg.get("models_root", models_root))
        except Exception:
            pass

    mm = ModelManager(models_root)
    services = Services(model_manager=mm)
    win = MainWindow(services=services)
    win.resize(1280, 800)
    win.show()

    # Auto model
    _auto_set_model(models_root)
    env_target = os.getenv("DEFAULT_MODEL_TARGET")
    if env_target:
        try:
            gen_mod.set_model_target(env_target)
            print(f"[AutoModel] Set model target from env: {env_target}", flush=True)
        except Exception as e:
            print(f"[AutoModel] Env target failed: {e}", flush=True)

    cache_dir = os.getenv("PIPELINE_CACHE_DIR")
    if cache_dir:
        try:
            win.statusBar().showMessage(f"Pipeline cache dir: {cache_dir}", 8000)
        except Exception:
            pass

    code = app.exec()
    try:
        from src.modules.utils import telemetry as _t
        if getattr(_t, "_FH", None):
            _t._FH.flush()
    except Exception:
        pass
    sys.exit(code)

if __name__ == "__main__":
    main()
