from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "gui.json"

_DEFAULT = {
    "remember_output_dir": True,
    "last_output_dir": "outputs/images",
    "last_model_path": "",
    "window_geometry": None,
    "window_state": None,
    "use_pipeline_disk_cache": True,
    "auto_release_enabled": False,
    "auto_release_minutes": 10,
    "auto_release_mode": "cpu",  # cpu or free
    "auto_save_to_gallery": True,
    "gallery_dir": r"C:\Users\beath\Documents\MyRepos\GitRepos\gallery",

    # Persisted UI state
    "last_prompt": "",
    "last_negative": "",
    "last_width": 1280,
    "last_height": 720,
    "last_steps": 30,
    "last_cfg": 7.5,
    "last_seed": 0,
    "last_batch": 1,
    "seed_auto_enabled": False,
    "last_sampler": "euler_a",
    "precision_mode": "FP32",
    "turbo_decode_mode": "FP32",
    "anatomy_guard_enabled": True,
    "auto_refine_enabled": False,
    "show_vram_on_start": False,
    "dark_mode": True,
}

def load_gui_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k, v in _DEFAULT.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return dict(_DEFAULT)

def save_gui_config(cfg: Dict[str, Any]) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass
