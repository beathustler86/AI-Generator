from __future__ import annotations
import json
import time
import os
from pathlib import Path
from threading import RLock

_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "telemetry.log"
_lock = RLock()

def sanitize_event(event):
	if isinstance(event, dict):
		return {k: str(v) if isinstance(v, os.PathLike) else v for k, v in event.items()}
	return event

def log_event(data: dict):
    """
    Lightweight append-only JSONL logger.
    """
    if not isinstance(data, dict):
        return
    rec = dict(data)
    rec.setdefault("ts", time.time())
    try:
        with _lock:
            with _LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def log_telemetry(**kwargs):
	entry = dict(kwargs)
	if "timestamp" not in entry:
		entry["timestamp"] = datetime.now().isoformat()
	log_event(entry)