from .collector import TelemetryCollector
from .config import telemetry_enabled
_collector = None

def ensure_telemetry_started():
    global _collector
    if not telemetry_enabled():
        return None
    if _collector is None:
        _collector = TelemetryCollector()
        import threading
        t = threading.Thread(target=_collector.run, name="TelemetryCollector", daemon=True)
        t.start()
    return _collector

def get_collector():
    return _collector