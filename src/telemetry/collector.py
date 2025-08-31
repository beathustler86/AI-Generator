import threading, time, json, os, datetime as dt, collections, csv
import psutil
from logging.handlers import RotatingFileHandler
from .config import telemetry_enabled, SAMPLE_HZ, LOG_DIR, RING_CAP, JSONL_MAX_MB, JSONL_BACKUPS
from .bus import event_q
from .nvml import gpu_stats

try:
    import torch
except Exception:
    torch = None

class TelemetryCollector:
    def __init__(self, hz=SAMPLE_HZ):
        self.hz = hz
        self.stop_evt = threading.Event()
        self.proc = psutil.Process()
        self.ring = collections.deque(maxlen=RING_CAP)
        os.makedirs(LOG_DIR, exist_ok=True)
        date = dt.datetime.utcnow().strftime("%Y%m%d")
        self.json_path = os.path.join(LOG_DIR, f"perf_{date}.jsonl")
        self.csv_path = os.path.join(LOG_DIR, "perf_latest.csv")
        self._csv_header = False
        self.logger = self._make_logger()
    def _make_logger(self):
        import logging
        log = logging.getLogger("perf")
        log.setLevel(logging.INFO)
        if not log.handlers:
            h = RotatingFileHandler(self.json_path, maxBytes=JSONL_MAX_MB*1024*1024, backupCount=JSONL_BACKUPS)
            h.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(h)
        return log
    def _torch_gpu(self):
        if not torch or not torch.cuda.is_available():
            return {}
        try:
            return {
                "torch_mem_alloc_mb": torch.cuda.memory_allocated()//1024//1024,
                "torch_mem_reserved_mb": torch.cuda.memory_reserved()//1024//1024,
            }
        except Exception:
            return {}
    def sample_once(self):
        vm = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        payload = {
            "type": "sample",
            "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds")+"Z",
            "cpu_sys_pct": psutil.cpu_percent(interval=None),
            "ram_used_mb": vm.used//1024//1024,
            "disk_read_mb": getattr(disk,"read_bytes",0)/1e6,
            "disk_write_mb": getattr(disk,"write_bytes",0)/1e6,
            "cpu_proc_pct": self.proc.cpu_percent(interval=None),
            "rss_mb": self.proc.memory_info().rss//1024//1024,
            **self._torch_gpu(),
            **gpu_stats()
        }
        self.ring.append(payload)
        self.logger.info(json.dumps(payload))
        self._csv(payload)
    def _csv(self, row):
        try:
            write_header = not self._csv_header
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    w.writeheader(); self._csv_header = True
                w.writerow(row)
        except Exception:
            pass
    def drain_events(self):
        while True:
            try:
                evt = event_q.get_nowait()
            except Exception:
                break
            else:
                self.logger.info(json.dumps(evt))
    def run(self):
        period = 1/max(1,self.hz)
        while not self.stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                self.sample_once()
                self.drain_events()
            except Exception as e:
                self.logger.info(json.dumps({"type":"collector_error","error":str(e)}))
            dt = period - (time.perf_counter()-t0)
            if dt>0: time.sleep(dt)
    def stop(self):
        self.stop_evt.set()
    def latest(self):
        return self.ring[-1] if self.ring else {}