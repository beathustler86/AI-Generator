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
import atexit

# Project root: telemetry.py is at <root>/src/modules/utils/telemetry.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = None
_FH = None

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
        # Ensure telemetry initialized
        if _LOG_FILE is None:
            init_telemetry()
        if _FH:
            line = json.dumps(rec, ensure_ascii=False)
            _FH.write(line + "\n")
            _FH.flush()
        else:
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

def init_telemetry():
    global _LOG_FILE, _FH
    if _LOG_FILE:
        return
    from pathlib import Path
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = log_dir / "telemetry.log"
    # Append instead of overwrite; line buffered
    _FH = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)
    atexit.register(_flush_telemetry)
    _emit({"event":"telemetry_init","file":str(_LOG_FILE)})

def _flush_telemetry():
    try:
        if _FH:
            _FH.flush()
    except Exception:
        pass

def _emit(obj: dict):
    try:
        import json
        obj = dict(obj)
        from datetime import datetime
        obj.setdefault("ts", datetime.utcnow().isoformat())
        obj.setdefault("level", "info")
        line = json.dumps(obj, ensure_ascii=False)
        print(line, flush=True)
        if _FH:
            _FH.write(line + "\n")
    except Exception:
        pass

__all__ = [
    "log_event",
    "log_telemetry",
    "log_exception",
    "init_telemetry",
]
