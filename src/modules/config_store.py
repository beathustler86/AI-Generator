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
    "auto_release_mode": "cpu"  # cpu or free
}

def load_gui_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k,v in _DEFAULT.items():
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
