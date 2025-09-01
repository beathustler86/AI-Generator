import sys
import traceback
from pathlib import Path
import json
import os

CURRENT = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT.parent
for p in (str(PROJECT_ROOT), str(CURRENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _run_preflight_blocking():
    try:
        from src.modules.preflight_check import run_preflight
    except Exception as e:
        print(f"[Preflight] Import error: {e}", flush=True)
        return
    try:
        result = run_preflight()
        with open(PROJECT_ROOT / "preflight_last.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        status = result.get("status")
        miss_req = len(result.get("missing_required", []))
        miss_opt = len(result.get("missing_optional", []))
        auto_created = len(result.get("auto_created_optional", []))
        print(
            f"[Preflight] status={status} "
            f"missing_required={miss_req} missing_optional={miss_opt} "
            f"auto_created={auto_created}", flush=True
        )
        if miss_req:
            print("[Preflight] Missing required files:", flush=True)
            for m in result["missing_required"]:
                print(f"  - {m}", flush=True)
    except Exception as e:
        print(f"[Preflight] Runtime error: {e}", flush=True)

_run_preflight_blocking()

try:
    from PySide6 import QtWidgets  # type: ignore
except ImportError as e:
    print("[FATAL] Qt binding import failed:", e)
    raise

try:
    from src.gui.main_window import MainWindow
    from src.gui.ui_setup import Services
    from src.modules.model_manager import ModelManager
except Exception as e:
    print("Failed to import MainWindow or dependencies:", e)
    traceback.print_exc()
    sys.exit(1)

DEFAULT_MODELS_ROOT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models"

def main():
    app = QtWidgets.QApplication(sys.argv)
    # Use env override if present
    models_root = Path(
        os.environ.get("MODELS_ROOT", DEFAULT_MODELS_ROOT)
    )
    import json, pathlib
    CFG_PATH = PROJECT_ROOT / "config" / "app.json"
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        models_root = cfg.get("models_root", models_root)
    mm = ModelManager(models_root)
    services = Services(model_manager=mm)
    win = MainWindow(services=services)
    win.resize(1280, 800)
    win.show()
    cache_dir = os.environ.get("PIPELINE_CACHE_DIR")
    if cache_dir:
        try:
            win.statusBar().showMessage(f"Pipeline cache dir: {cache_dir}", 8000)
        except Exception:
            pass
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

def on_open_models_dir(self) -> None:
    mm = getattr(self.services, "model_manager", None)
    path = getattr(mm, "models_root", None)
    if not path:
        self._status("No models root configured.")
        return
    from pathlib import Path
    if not Path(path).exists():
        self._status(f"Models root missing: {path}")
        return
    # open logic unchanged...
