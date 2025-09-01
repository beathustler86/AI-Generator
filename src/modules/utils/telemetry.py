from __future__ import annotations
import json
import time
import os
import sys
import traceback
from pathlib import Path
from threading import RLock
from datetime import datetime
from typing import Any, Optional, Dict

# Project root: telemetry.py is at <root>/src/modules/utils/telemetry.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "telemetry.log"

_lock = RLock()

def _serialize(obj: Any):
    try:
        return str(obj)
    except Exception:
        return "<unrepr>"

def sanitize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in event.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, os.PathLike):
            clean[k] = str(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [_serialize(x) for x in v]
        elif isinstance(v, dict):
            clean[k] = {sk: _serialize(sv) for sk, sv in v.items()}
        else:
            clean[k] = _serialize(v)
    return clean

def _write_line(payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with _lock:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

def log_event(data: Dict[str, Any], level: str = "info") -> None:
    """
    Append an event (JSON line) to project_root/logs/telemetry.log
    """
    if not isinstance(data, dict):
        return
    rec = sanitize_event(data)
    rec.setdefault("ts", datetime.utcnow().isoformat())
    rec.setdefault("level", level)
    try:
        _write_line(rec)
    except Exception:
        pass

def log_telemetry(**kwargs):
    if "timestamp" not in kwargs:
        kwargs["timestamp"] = datetime.utcnow().isoformat()
    log_event(kwargs)

def log_exception(exc: BaseException, context: Optional[str] = None):
    log_event({
        "event": "exception",
        "context": context,
        "type": type(exc).__name__,
        "message": str(exc),
        "trace": "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
    }, level="error")

def install_global_exception_logger():
    orig_hook = sys.excepthook
    def _hook(exc_type, exc_val, exc_tb):
        try:
            log_event({
                "event": "uncaught",
                "type": exc_type.__name__,
                "message": str(exc_val),
                "trace": "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            }, level="error")
        finally:
            orig_hook(exc_type, exc_val, exc_tb)
    sys.excepthook = _hook

def install_threading_excepthook():
    import threading
    if hasattr(threading, "excepthook"):
        orig = threading.excepthook
        def _thook(args):
            try:
                log_event({
                    "event": "thread_exception",
                    "thread_name": args.thread.name,
                    "type": args.exc_type.__name__,
                    "message": str(args.exc_value),
                    "trace": "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
                }, level="error")
            finally:
                orig(args)
        threading.excepthook = _thook

def init_telemetry(global_ex: bool = True, thread_ex: bool = True):
    if global_ex:
        install_global_exception_logger()
    if thread_ex:
        install_threading_excepthook()
    log_event({"event": "telemetry_init", "file": str(_LOG_FILE)})

__all__ = [
    "log_event",
    "log_telemetry",
    "log_exception",
    "init_telemetry",
]
