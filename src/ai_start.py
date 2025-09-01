import os, sys, subprocess, importlib.util, pathlib, json, time, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LAUNCH = SRC / "launch_gui.py"
DEV_CHECK_DEPS = os.environ.get("DEV_CHECK_DEPS", "0") == "1"
DEV_FORCE_DEPS = os.environ.get("DEV_FORCE_DEPS", "0") == "1"
CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

TORCH_PKGS = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "torchaudio": "2.6.0+cu124",
}

# Include PySide6 here so we do not duplicate logic inside launch_gui.py
CORE_PKGS = [
    "PySide6",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow",
]

def _need(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None

def _install(cmd_args):
    print(f"[Deps] pip {' '.join(cmd_args[2:])}")
    subprocess.check_call(cmd_args)

def ensure_torch():
    missing = [m for m in TORCH_PKGS if _need(m)]
    if missing or DEV_FORCE_DEPS:
        print(f"[Deps] Installing torch stack (CUDA 12.4) missing={missing}")
        _install([
            sys.executable, "-m", "pip", "install",
            "--extra-index-url", CUDA_INDEX,
            *(f"{k}=={v}" for k, v in TORCH_PKGS.items())
        ])
    else:
        print("[Deps] Torch OK")

def ensure_core():
    missing = [m for m in CORE_PKGS if _need(m)]
    if missing or DEV_FORCE_DEPS:
        print(f"[Deps] Installing core packages missing={missing}")
        _install([sys.executable, "-m", "pip", "install", *CORE_PKGS])
    else:
        print("[Deps] Core OK")

def preflight_once():
    try:
        from src.modules.preflight_check import run_preflight
    except Exception as e:
        print(f"[Preflight] Import fail: {e}")
        return
    try:
        res = run_preflight()
        with open(ROOT / "preflight_last.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        status = res.get("status")
        miss_req = len(res.get("missing_required", []))
        miss_opt = len(res.get("missing_optional", []))
        print(f"[Preflight] status={status} missing_required={miss_req} missing_optional={miss_opt}")
        if miss_req:
            for m in res["missing_required"]:
                print("  -", m)
    except Exception:
        print("[Preflight] Error:")
        traceback.print_exc()

def _assert_qt():
    if _need("PySide6"):
        print("[FATAL] PySide6 not installed. Run with DEV_CHECK_DEPS=1 or install manually: pip install PySide6")
        sys.exit(3)

def main():
    t0 = time.time()
    if DEV_CHECK_DEPS or DEV_FORCE_DEPS:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--disable-pip-version-check", "--upgrade", "pip"])
        except Exception as e:
            print(f"[Deps] pip upgrade failed (continuing): {e}")
        ensure_torch()
        ensure_core()
    else:
        print("[Deps] Skipped dependency auto-install (set DEV_CHECK_DEPS=1 to enable)")

    _assert_qt()
    preflight_once()

    if not LAUNCH.exists():
        print(f"[Launcher] launch_gui.py missing at {LAUNCH}", file=sys.stderr)
        return 2

    print(f"[Launcher] Starting GUI (init {int((time.time()-t0)*1000)}ms)")
    glb = {"__name__": "__main__"}
    code = compile(LAUNCH.read_text(encoding="utf-8"), str(LAUNCH), "exec")
    exec(code, glb)
    return 0

if __name__ == "__main__":
    sys.exit(main())
